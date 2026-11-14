import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch

from kaggle_student import evaluate, find_file, restore_run
from kaggle_train import find_root, run, take_lines, train_until_done

# The full proposed model and its controlled comparisons, on one frozen split, in two Kaggle sessions:
#   --stage teachers   the original CHAL teacher and the FDAM teacher, same settings, then both scored
#   --stage students   S0 labels only, S1 feature distillation (original teacher), S2 frequency-aligned distillation
#                      (original teacher), S3 the full model: frequency-aligned distillation from the FDAM teacher
# Each stage picks one epoch count for all its models from measured speed, so they fit in --max-hours. Run a stage
# again with its earlier output attached and it skips finished models and resumes the rest with the same plan.

# Validation: 8 sequences of train.txt, including two of its three low-contrast ones (44, 75). Test: test.txt unchanged.
FROZEN_VAL = 'Sequence1,Sequence12,Sequence18,Sequence26,Sequence38,Sequence44,Sequence75,Sequence82'

TEACHERS = [('teacher_base', '0'), ('teacher_fdam', '1')]
STUDENTS = [('S0_labels', 'labels', None), ('S1_feature', 'feature', 'teacher_base'),
            ('S2_frequency', 'frequency', 'teacher_base'), ('S3_full', 'frequency', 'teacher_fdam')]

def find_in(input_dir, tag, name, unique=False):
    found = []
    for dir_path, _, files in os.walk(input_dir, followlinks=True):
        if name in files and tag in dir_path.split(os.sep):
            found.append(os.path.join(dir_path, name))
    if unique and len(found) > 1:
        raise RuntimeError('Several %s/%s under the inputs, attach only the newest output: %s' % (tag, name, found))
    return found[0] if found else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['teachers', 'students'], required=True)
    parser.add_argument('--input', default='/kaggle/input')
    parser.add_argument('--work', default='/kaggle/working')
    parser.add_argument('--code-dir', default='/tmp/CHAL')
    parser.add_argument('--max-hours', type=float, default=11.0)
    parser.add_argument('--max-epochs', type=int, default=0, help='cap on the planned epochs; 0 = 15 for teachers, 17 for students')
    parser.add_argument('--min-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--save-period', type=int, default=5)
    parser.add_argument('--restart-every', type=int, default=5, help='epochs per training process')
    parser.add_argument('--no-fp16', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--bg-strength', type=float, default=25.0)
    parser.add_argument('--speed-iters', type=int, default=50)
    parser.add_argument('--score-hours', type=float, default=0.0, help='time set aside to score each model; 0 = 0.75 teacher, 0.4 student')
    parser.add_argument('--limit-train', type=int, default=0, help='use only this many frames, for trial runs')
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--redo', default='', help='comma-separated models to train again from scratch, ignoring earlier outputs')
    args = parser.parse_args()
    redo = set(filter(None, args.redo.split(',')))
    started = time.time()
    teachers = args.stage == 'teachers'
    score_hours = args.score_hours or (0.75 if teachers else 0.4)
    max_epochs = args.max_epochs or (15 if teachers else 17)

    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or 'none', flush=True)
    data_root = find_root(args.input, 'train.txt', 'test.txt', 'Sequence1')
    source = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.code_dir):
        shutil.copytree(source, args.code_dir, ignore=shutil.ignore_patterns('.git', 'logs*', 'datasets', '__pycache__'))
    code = args.code_dir
    annotations = os.path.join(code, 'datasets', 'NUDT-MIRSDT')
    subprocess.check_call([sys.executable, os.path.join('utils_coco', 'nudt_to_coco.py'), '--root', data_root, '--out', annotations,
                           '--val-list', FROZEN_VAL], cwd=code)
    os.makedirs(args.work, exist_ok=True)
    shutil.copy(os.path.join(annotations, 'split_manifest.json'), os.path.join(args.work, 'split_manifest.json'))

    train_txt = os.path.join(annotations, 'coco_train_NUDT.txt')
    val_txt = os.path.join(annotations, 'coco_val_NUDT.txt')
    run_train = take_lines(train_txt, os.path.join(annotations, 'run_train.txt'), args.limit_train)
    run_val = take_lines(val_txt, os.path.join(annotations, 'run_val.txt'), args.limit_val)
    with open(run_train) as f:
        iters_per_epoch = sum(1 for _ in f) // args.batch_size // 2
    base_env = {
        'CHAL_BATCH_SIZE': str(args.batch_size), 'CHAL_NUM_WORKERS': str(args.num_workers),
        'CHAL_FP16': '0' if args.no_fp16 or not torch.cuda.is_available() else '1',
        'CHAL_TRAIN_ANNOTATION': run_train, 'CHAL_VAL_ANNOTATION': run_val, 'CHAL_BG_STRENGTH': str(args.bg_strength),
    }
    splits = [('val', 'val', 0), ('test', 'test', 0), ('test_bg', 'test', args.bg_strength)]

    # What each model needs: (tag, training script, its environment, what test.py needs to load it).
    models = []
    if teachers:
        for tag, fdam in TEACHERS:
            models.append((tag, 'train.py', dict(base_env, CHAL_FDAM=fdam), {'CHAL_MODEL_KIND': 'teacher', 'CHAL_FDAM': fdam}))
    else:
        teacher_paths = {}
        for tag, _ in TEACHERS:
            teacher_paths[tag] = find_in(args.input, tag, 'teacher_ema.pth', unique=True)
            print('%s: %s' % (tag, teacher_paths[tag]), flush=True)
        for tag, mode, teacher in STUDENTS:
            env = dict(base_env, CHAL_STUDENT_MODE=mode, CHAL_STUDENT_WIDTH='0.25')
            if teacher:
                if not teacher_paths[teacher]:
                    raise FileNotFoundError('No %s/teacher_ema.pth under %s. Attach the teachers stage output.' % (teacher, args.input))
                env.update(CHAL_TEACHER=teacher_paths[teacher], CHAL_TEACHER_FDAM='1' if teacher == 'teacher_fdam' else '0')
            models.append((tag, 'train_student.py', env, {'CHAL_MODEL_KIND': 'student', 'CHAL_STUDENT_WIDTH': '0.25'}))

    def measure_speed(script, env, name):
        scratch = os.path.join(code, 'speed_' + name)
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        rate = run(script, code, dict(env,
            CHAL_TRAIN_ANNOTATION=take_lines(train_txt, os.path.join(scratch, 'train.txt'), args.speed_iters * args.batch_size * 2),
            CHAL_VAL_ANNOTATION=take_lines(val_txt, os.path.join(scratch, 'val.txt'), args.batch_size * 2),
            CHAL_SAVE_DIR=os.path.join(scratch, 'logs'), CHAL_EPOCHS='1', CHAL_EVAL_PERIOD='9999', CHAL_SAVE_PERIOD='9999'))
        shutil.rmtree(scratch, ignore_errors=True)
        if not rate:
            raise RuntimeError('Could not read the training speed of %s from its progress bar.' % name)
        return rate

    # The plan is fixed the first time and reused when a stage is run again, so every model gets the same epochs.
    plan_name = 'plan_%s.json' % args.stage
    earlier_plan = find_file(args.input, plan_name) if any(plan_name in files for _, _, files in os.walk(args.input, followlinks=True)) else None
    if earlier_plan:
        with open(earlier_plan) as f:
            plan = json.load(f)
        print('Using the earlier plan from %s: %d epochs.' % (earlier_plan, plan['epochs']), flush=True)
    else:
        rates = {}
        for tag, script, env, _ in models:
            # Models of one kind train at the same speed, so one measurement per kind is enough.
            kind = (script, env.get('CHAL_STUDENT_MODE', 'teacher') == 'labels', env.get('CHAL_FDAM', '0'))
            if kind not in rates:
                rates[kind] = measure_speed(script, env, tag)
            rates[tag] = rates[kind]
        epoch_hours = {tag: iters_per_epoch / rates[tag] / 3600 * 1.2 for tag, _, _, _ in models}
        left = args.max_hours - (time.time() - started) / 3600 - len(models) * score_hours - 0.25
        epochs = min(max_epochs, max(args.min_epochs, int(left / sum(epoch_hours.values()))))
        plan = {'stage': args.stage, 'epochs': epochs, 'iters_per_epoch': iters_per_epoch, 'epoch_hours': epoch_hours,
                'speed_it_per_s': {tag: rates[tag] for tag, _, _, _ in models}, 'frozen_val': FROZEN_VAL, 'seed': 3407}
        print('Plan: %d epochs for each of %s, about %.1f hours of training.' % (
            epochs, ', '.join(tag for tag, _, _, _ in models), epochs * sum(epoch_hours.values())), flush=True)
    with open(os.path.join(args.work, plan_name), 'w') as f:
        json.dump(plan, f, indent=2)
    epochs = plan['epochs']

    summary_path = os.path.join(args.work, 'summary_%s.json' % args.stage)
    summary = {'plan': plan, 'models': {}}
    for i, (tag, script, env, load_env) in enumerate(models):
        out = os.path.join(args.work, tag)
        done = None if tag in redo else find_in(args.input, tag, 'results.json')
        if done:
            print('\n===== %s: already finished in an earlier session, copying it =====' % tag, flush=True)
            shutil.copytree(os.path.dirname(done), out, dirs_exist_ok=True)
        else:
            later = sum(epochs * plan['epoch_hours'][t] + score_hours for t, _, _, _ in models[i + 1:])
            hours = args.max_hours - (time.time() - started) / 3600 - later - score_hours
            print('\n===== %s: %d epochs, up to %.1f hours of training =====' % (tag, epochs, hours), flush=True)
            try:
                save_dir = os.path.join(out, 'logs')
                resume_path = os.path.join(save_dir, 'resume.pth')
                os.makedirs(save_dir, exist_ok=True)
                if tag not in redo:
                    restore_run(args.input, tag, resume_path)
                run_env = dict(env, CHAL_SAVE_DIR=save_dir, CHAL_EPOCHS=str(epochs), CHAL_EVAL_PERIOD='9999', CHAL_SAVE_PERIOD=str(args.save_period))
                with open(os.path.join(out, 'config.json'), 'w') as f:
                    json.dump({'script': script, 'env': run_env, 'plan': plan}, f, indent=2)
                run_args = argparse.Namespace(**vars(args))
                run_args.epochs, run_args.max_hours = epochs, max(hours, 0.05)
                finished = train_until_done(code, run_env, resume_path, run_args, script)
                print('%s: finished %d of %d epochs.' % (tag, finished, epochs), flush=True)
                if finished == 0:
                    raise RuntimeError('no epoch finished')
                weights = os.path.join(out, 'teacher_ema.pth' if teachers else 'student_ema.pth')
                torch.save(torch.load(resume_path, map_location='cpu')['ema'], weights)
                evaluate(code, args, data_root, annotations, out, dict(load_env, CHAL_MODEL_PATH=weights), splits)
                with open(os.path.join(out, 'results.json')) as f:
                    results = json.load(f)
                results['finished_epochs'], results['planned_epochs'] = finished, epochs
                with open(os.path.join(out, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                if finished < epochs:
                    # Unfinished: its scores are kept for reference, but a rerun should train it again from its checkpoint.
                    os.rename(os.path.join(out, 'results.json'), os.path.join(out, 'results_unfinished.json'))
            except Exception as error:
                print('%s failed: %s. Going on with the next model.' % (tag, error), flush=True)
                summary['models'][tag] = {'error': str(error)}
        for name in ('results.json', 'results_unfinished.json'):
            path = os.path.join(out, name)
            if os.path.exists(path):
                with open(path) as f:
                    summary['models'][tag] = json.load(f)
                with open(os.path.join(out, 'report_test.json')) as f:
                    report = json.load(f)
                summary['models'][tag]['test_low_contrast'] = report['low_contrast']
                summary['models'][tag]['test_other'] = report['other']
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n===== Summary, test set =====')
    print('%-14s %7s %6s %6s %7s %7s %10s %8s' % ('model', 'epochs', 'Pd', 'F1', 'FA/fr', 'mAP50', 'Pd low-c', 'Pd bg'))
    for tag, result in summary['models'].items():
        if 'test' in result:
            t, low, bg = result['test'], result.get('test_low_contrast') or {}, result['test_bg']
            print('%-14s %7s %6.3f %6.3f %7.3f %7.3f %10.3f %8.3f' % (tag, '%d/%d' % (result['finished_epochs'], result['planned_epochs']),
                  t['pd'], t['f1'], t['false_alarms_per_frame'], t['map50'], low.get('pd', float('nan')), bg['pd']))
        else:
            print('%-14s failed: %s' % (tag, result.get('error')))
    print('Saved to', summary_path)
