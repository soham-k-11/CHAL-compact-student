import copy
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.distill import DistillLoss, TeacherResponses
from nets.slowfastnet import slowfastnet
from nets.student import CompactStudent
from nets.yolo_training import ModelEMA, YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from train import memory_report
from utils.background_change import background_change
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader_for_DAUB import dataset_collate, seqDataset
from utils.utils import get_classes, get_lr, show_config

# Kinds of run, for the comparison in the report:
#   labels     the student alone on the true labels, the baseline
#   feature    labels plus conventional feature distillation (plain squared error on the teacher's final features)
#   frequency  labels plus frequency-aligned feature distillation (the same error, low and high bands weighted apart)
#   distill    labels plus matching the teacher's predictions, anomaly map and features
#   robust     distill, also on background-changed clips, plus staying the same under those changes

def to_device(responses, device):
    return {key: value.to(device) for key, value in responses.items()}

def recalibrate_batchnorm(model, loader, device, steps):
    # The running statistics trail the weights during training; re-estimate them once at the end as a plain average
    # over training batches, so the saved model behaves in evaluation mode as it did in training.
    layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momentum = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    with torch.no_grad():
        for step, (images, _) in enumerate(loader):
            if step >= steps:
                break
            model(images.to(device))
    for m, value in zip(layers, momentum):
        m.momentum = value
    model.eval()

if __name__ == "__main__":
    run_mode        = os.environ.get('CHAL_STUDENT_MODE', 'distill')
    width           = float(os.environ.get('CHAL_STUDENT_WIDTH', 0.25))
    teacher_path    = os.environ.get('CHAL_TEACHER', 'model_data/teacher_ema.pth')
    epochs          = int(os.environ.get('CHAL_EPOCHS', 30))
    batch_size      = int(os.environ.get('CHAL_BATCH_SIZE', 4))
    fp16            = os.environ.get('CHAL_FP16', '0') == '1'
    init_lr         = float(os.environ.get('CHAL_LR', 5e-4))
    distill_weight  = float(os.environ.get('CHAL_DISTILL_WEIGHT', 1.0))
    invariance_weight = float(os.environ.get('CHAL_INVARIANCE_WEIGHT', 1.0))
    bg_strength     = float(os.environ.get('CHAL_BG_STRENGTH', 25.0))
    freq_cutoff     = float(os.environ.get('CHAL_FREQ_CUTOFF', 0.25))
    freq_high_weight = float(os.environ.get('CHAL_FREQ_HIGH_WEIGHT', 2.0))
    freq_rule       = os.environ.get('CHAL_FREQ_RULE', 'balanced')
    save_period     = int(os.environ.get('CHAL_SAVE_PERIOD', 5))
    eval_period     = int(os.environ.get('CHAL_EVAL_PERIOD', 10))
    num_workers     = int(os.environ.get('CHAL_NUM_WORKERS', 2))
    save_dir        = os.environ.get('CHAL_SAVE_DIR', 'logs_student')
    max_hours       = float(os.environ.get('CHAL_MAX_HOURS', 0))
    run_epochs      = int(os.environ.get('CHAL_RUN_EPOCHS', 0))
    max_iters       = int(os.environ.get('CHAL_MAX_ITERS', 0))
    train_path      = os.environ.get('CHAL_TRAIN_ANNOTATION', 'datasets/NUDT-MIRSDT/coco_train_NUDT.txt')
    val_path        = os.environ.get('CHAL_VAL_ANNOTATION', 'datasets/NUDT-MIRSDT/coco_val_NUDT.txt')
    input_shape     = [512, 512]
    resume_path     = os.path.join(save_dir, 'resume.pth')
    start_time      = time.time()
    seed            = 3407

    if run_mode not in ('labels', 'feature', 'frequency', 'distill', 'robust'):
        raise ValueError('Unknown CHAL_STUDENT_MODE %s' % run_mode)

    # With two GPUs the teacher gets its own, as it only runs forward and has hooks that do not suit DataParallel.
    Cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if Cuda else 'cpu')
    teacher_device = torch.device('cuda:1' if torch.cuda.device_count() > 1 else device)

    os.makedirs(save_dir, exist_ok=True)
    class_names, num_classes = get_classes('model_data/classes.txt')

    def set_seed(value):
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        np.random.seed(value)
        random.seed(value)
    set_seed(seed)

    model = CompactStudent(num_classes, num_frame=5, width=width)
    if os.environ.get('CHAL_INIT', 'default') == 'normal':
        weights_init(model)
    if os.environ.get('CHAL_PRIOR', '1') == '1':
        model.init_prior()
    model = model.to(device)
    distill = DistillLoss(model.feature_dim).to(device)

    teacher = None
    if run_mode != 'labels':
        network = slowfastnet(num_classes, num_frame=5, fdam=os.environ.get('CHAL_TEACHER_FDAM', '0') == '1')
        network.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        teacher = TeacherResponses(network.to(teacher_device))

    yolo_loss = YOLOLoss(num_classes, fp16, strides=[8])
    log_dir = os.path.join(save_dir, 'loss_' + datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S'))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    scaler = None
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    ema = ModelEMA(model)

    with open(train_path, encoding='utf-8') as f:
        num_train = len(f.readlines())
    with open(val_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    show_config(run_mode=run_mode, width=width, teacher_path=teacher_path, epochs=epochs, batch_size=batch_size, init_lr=init_lr,
                distill_weight=distill_weight, invariance_weight=invariance_weight, bg_strength=bg_strength,
                freq_cutoff=freq_cutoff, freq_high_weight=freq_high_weight, freq_rule=freq_rule,
                teacher_fdam=os.environ.get('CHAL_TEACHER_FDAM', '0'), seed=seed, save_dir=save_dir,
                params=sum(p.numel() for p in model.parameters()), device=device, teacher_device=teacher_device, num_train=num_train)

    decay, no_decay = [], []
    for name, p in list(model.named_parameters()) + list(distill.named_parameters()):
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = optim.AdamW([{'params': decay, 'weight_decay': 5e-2}, {'params': no_decay, 'weight_decay': 0.0}], init_lr, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler_func = get_lr_scheduler('cos', init_lr, init_lr * 0.01, epochs)

    init_epoch = 0
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        distill.load_state_dict(checkpoint['distill'])
        ema.ema.load_state_dict(checkpoint['ema'])
        ema.updates = checkpoint['ema_updates']
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        init_epoch = checkpoint['epoch'] + 1
        print('Resumed from {} at epoch {}.'.format(resume_path, init_epoch))
        # A restarted process must not replay the shuffle order of the first run.
        set_seed(seed + init_epoch)
    field_generator = torch.Generator().manual_seed(seed + init_epoch)

    train_dataset = seqDataset(train_path, input_shape[0], 5, 'train', None)
    val_dataset = seqDataset(val_path, input_shape[0], 5, 'val', None)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=dataset_collate)
    # As in the teacher's training, one epoch uses half of the frames.
    epoch_step = num_train // batch_size // 2
    if max_iters:
        epoch_step = min(epoch_step, max_iters)
    epoch_step_val = len(val_lines) // batch_size
    if max_iters:
        epoch_step_val = min(epoch_step_val, max_iters)
    eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, eval_flag=True, period=eval_period)

    def clip_losses(images, targets):
        student = model(images, return_responses=True)
        # The detection loss writes decoded boxes into the tensor it is given, so it gets its own copy.
        losses = {'det': yolo_loss([student['prediction'].clone()], targets)}
        if teacher is not None:
            reference = to_device(teacher(images.to(teacher_device)), device)
            if run_mode == 'feature':
                losses['feature'] = distill.feature_loss(student, reference)
            elif run_mode == 'frequency':
                losses['frequency'] = distill.frequency_loss(student, reference, freq_cutoff, freq_high_weight, freq_rule)
            else:
                for key, value in distill(student, reference).items():
                    losses[key] = value
        return student, losses

    def total_of(losses):
        return losses['det'] + distill_weight * sum(value for key, value in losses.items() if key != 'det')

    def training_loss(images, targets):
        student, losses = clip_losses(images, targets)
        total = total_of(losses)
        if run_mode == 'robust':
            changed_images = background_change(images, [t.cpu().numpy() for t in targets], bg_strength, generator=field_generator)
            changed, changed_losses = clip_losses(changed_images, targets)
            invariance = distill.invariance_loss(changed, student)
            total = 0.5 * (total + total_of(changed_losses)) + invariance_weight * invariance
            losses['invariance'] = invariance
        return total, losses

    def validation_loss(net):
        net.eval()
        total = 0.0
        for iteration, (images, targets) in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            with torch.no_grad():
                total += yolo_loss(net(images.to(device)), [t.to(device) for t in targets]).item()
        return total / epoch_step_val

    for epoch in range(init_epoch, epochs):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        model.train()
        totals = {}
        loss_sum = 0.0
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{epochs}', mininterval=0.3)
        for iteration, (images, targets) in enumerate(gen):
            if iteration >= epoch_step:
                break
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            optimizer.zero_grad()
            if scaler is None:
                loss, parts = training_loss(images, targets)
                loss.backward()
                optimizer.step()
            else:
                with torch.cuda.amp.autocast():
                    loss, parts = training_loss(images, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            ema.update(model)
            loss_sum += loss.item()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            pbar.set_postfix(loss=loss_sum / (iteration + 1), lr=get_lr(optimizer))
            pbar.update(1)
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        val_loss = validation_loss(ema.ema)
        if epoch + 1 == epochs:
            # Re-estimated statistics are kept only if they do not make the validation loss worse.
            before = copy.deepcopy(ema.ema.state_dict())
            recalibrate_batchnorm(ema.ema, gen, device, min(200, epoch_step))
            recalibrated = validation_loss(ema.ema)
            print('Batch-norm statistics re-estimated: validation loss %.3f before, %.3f after, keeping %s.' % (
                val_loss, recalibrated, 'after' if recalibrated <= val_loss else 'before'))
            if recalibrated <= val_loss:
                val_loss = recalibrated
            else:
                ema.ema.load_state_dict(before)
        print('Finish Validation')

        train_loss = loss_sum / epoch_step
        loss_history.append_loss(epoch + 1, train_loss, val_loss)
        eval_callback.on_epoch_end(epoch + 1, ema.ema)
        print('Epoch:%d/%d' % (epoch + 1, epochs))
        print('Total Loss: %.3f || Val Loss: %.3f' % (train_loss, val_loss))
        print('Parts: ' + ', '.join('%s %.3f' % (key, value / epoch_step) for key, value in totals.items()))

        if (epoch + 1) % save_period == 0 or epoch + 1 == epochs:
            torch.save(ema.ema.state_dict(), os.path.join(save_dir, 'ep%03d.pth' % (epoch + 1)))
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'distill': distill.state_dict(),
            'ema': ema.ema.state_dict(),
            'ema_updates': ema.updates,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
        }, resume_path + '.tmp')
        os.replace(resume_path + '.tmp', resume_path)
        print('Memory after epoch {}: {}'.format(epoch + 1, memory_report()))

        if max_hours > 0 and (time.time() - start_time) / 3600 >= max_hours:
            print('Reached CHAL_MAX_HOURS after epoch {}, stopping.'.format(epoch + 1))
            break
        if run_epochs > 0 and epoch + 1 - init_epoch >= run_epochs:
            print('Ran {} epochs in this process, stopping for a restart.'.format(run_epochs))
            break

    loss_history.writer.close()
