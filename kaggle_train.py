import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

# Runs the CHAL teacher on Kaggle: speed test, time-boxed training that resumes across sessions, then evaluation.
# Kaggle's input folders are read-only and a session stops at 12 hours, so the code runs from a temporary copy
# and everything worth keeping goes to the output folder.

RATE = re.compile(r'([\d.]+)\s*(it/s|s/it)')

class ProcessKilled(RuntimeError):
    pass

def find_root(input_dir, *required):
    for dir_path, dirs, files in os.walk(input_dir, followlinks=True):
        if all(name in files or name in dirs for name in required):
            return dir_path
    raise FileNotFoundError('No folder with %s under %s. Is the dataset attached?' % (', '.join(required), input_dir))

def run(script, cwd, env):
    process = subprocess.Popen([sys.executable, script], cwd=cwd, env={**os.environ, **env}, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    rate, in_train, bars = None, False, 0
    for line in process.stdout:
        line = line.rstrip()
        match = RATE.search(line)
        if match:
            if in_train:
                value = float(match.group(1))
                rate = value if match.group(2) == 'it/s' else 1 / value
            bars += 1
            if bars % 500 == 0:
                print(line, flush=True)
            continue
        print(line, flush=True)
        if 'Start Train' in line:
            in_train = True
        elif 'Finish Train' in line:
            in_train = False
        elif 'Total Loss' in line and 'nan' in line:
            process.kill()
            raise RuntimeError('Training loss became NaN. Try running again with --no-fp16.')
    if process.wait() == -9:
        raise ProcessKilled('%s was killed by the system, most likely out of RAM' % script)
    if process.returncode != 0:
        raise RuntimeError('%s failed with exit code %d' % (script, process.returncode))
    return rate

def finished_epochs(resume_path):
    if not os.path.exists(resume_path):
        return 0
    return torch.load(resume_path, map_location='cpu')['epoch'] + 1

def train_until_done(code, env, resume_path, args, script='train.py'):
    # Each process runs a few epochs and exits, so any slow memory growth starts again from zero.
    deadline = time.time() + args.max_hours * 3600
    failures = 0
    while True:
        before = finished_epochs(resume_path)
        remaining = (deadline - time.time()) / 3600
        if before >= args.epochs or remaining <= 0:
            return before
        env['CHAL_MAX_HOURS'] = str(remaining)
        env['CHAL_RUN_EPOCHS'] = str(args.restart_every)
        try:
            run(script, code, env)
        except ProcessKilled as error:
            failures = failures + 1 if finished_epochs(resume_path) == before else 1
            print('%s (%d in a row without a finished epoch).' % (error, failures), flush=True)
            if failures >= 3:
                print('Giving up, training cannot finish an epoch.', flush=True)
                return finished_epochs(resume_path)

def take_lines(src, dst, count):
    with open(src) as f:
        lines = f.readlines()
    with open(dst, 'w') as f:
        f.writelines(lines[:count] if count else lines)
    return dst

def take_frames(src, dst, count):
    with open(src) as f:
        data = json.load(f)
    if count:
        data['images'] = data['images'][:count]
        kept = {image['id'] for image in data['images']}
        data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in kept]
    with open(dst, 'w') as f:
        json.dump(data, f)
    return dst

def restore_checkpoint(input_dir, resume_path):
    best_epoch, best_path = -1, None
    if os.path.exists(resume_path):
        return
    for dir_path, _, files in os.walk(input_dir, followlinks=True):
        if 'resume.pth' in files:
            path = os.path.join(dir_path, 'resume.pth')
            epoch = torch.load(path, map_location='cpu')['epoch']
            if epoch > best_epoch:
                best_epoch, best_path = epoch, path
    if best_path:
        os.makedirs(os.path.dirname(resume_path), exist_ok=True)
        shutil.copy(best_path, resume_path)
        print('Continuing from %s, which finished epoch %d.' % (best_path, best_epoch + 1))

def evaluate(code, args, data_root, teacher, annotations):
    results = {}
    threshold = None
    for split in ('val', 'test'):
        gt = take_frames(os.path.join(annotations, 'instances_%s.json' % split),
                         os.path.join(annotations, 'eval_%s.json' % split), args.limit_eval)
        env = {'CHAL_MODEL_PATH': teacher, 'CHAL_COCO_GT': gt, 'CHAL_DATASET_IMG_PATH': data_root,
               'CHAL_INPUT_SIZE': '512', 'CHAL_LETTERBOX': '1'}
        # The threshold is picked on validation and only then applied to the test set.
        if threshold is not None:
            env['CHAL_CONF_THRES'] = str(threshold)
        run('test.py', code, env)
        with open(os.path.join(code, 'map_out', 'coco_eval', 'metrics.json')) as f:
            results[split] = json.load(f)
        threshold = results[split]['threshold']
    path = os.path.join(args.work, 'teacher_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Test results:', json.dumps(results['test'], indent=2))
    print('Saved to', path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['speed', 'train'], required=True)
    parser.add_argument('--input', default='/kaggle/input')
    parser.add_argument('--work', default='/kaggle/working')
    parser.add_argument('--code-dir', default='/tmp/CHAL')
    parser.add_argument('--epochs', type=int, default=30, help='keep the same across sessions, it sets the learning-rate curve')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-hours', type=float, default=10.5, help='stop after the epoch that passes this')
    parser.add_argument('--eval-period', type=int, default=10)
    parser.add_argument('--save-period', type=int, default=5)
    parser.add_argument('--restart-every', type=int, default=5, help='epochs per training process')
    parser.add_argument('--no-fp16', action='store_true')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--speed-iters', type=int, default=50)
    parser.add_argument('--limit-train', type=int, default=0, help='use only this many frames, for trial runs')
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    args = parser.parse_args()

    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or 'none')
    data_root = find_root(args.input, 'train.txt', 'test.txt', 'Sequence1')
    source = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.code_dir):
        shutil.copytree(source, args.code_dir, ignore=shutil.ignore_patterns('.git', 'logs', 'datasets', '__pycache__'))
    code = args.code_dir
    annotations = os.path.join(code, 'datasets', 'NUDT-MIRSDT')
    subprocess.check_call([sys.executable, os.path.join('utils_coco', 'nudt_to_coco.py'), '--root', data_root, '--out', annotations], cwd=code)

    train_txt = os.path.join(annotations, 'coco_train_NUDT.txt')
    val_txt = os.path.join(annotations, 'coco_val_NUDT.txt')
    with open(train_txt) as f:
        num_train = sum(1 for _ in f)
    iters_per_epoch = (num_train // args.batch_size) // 2

    env = {
        'CHAL_BATCH_SIZE': str(args.batch_size),
        'CHAL_NUM_WORKERS': str(args.num_workers),
        'CHAL_FP16': '0' if args.no_fp16 or not torch.cuda.is_available() else '1',
    }

    if args.mode == 'speed':
        scratch = os.path.join(code, 'speed')
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        lines = args.speed_iters * args.batch_size * 2
        env.update({
            'CHAL_TRAIN_ANNOTATION': take_lines(train_txt, os.path.join(scratch, 'train.txt'), lines),
            'CHAL_VAL_ANNOTATION': take_lines(val_txt, os.path.join(scratch, 'val.txt'), args.batch_size * 2),
            'CHAL_SAVE_DIR': os.path.join(scratch, 'logs'), 'CHAL_EPOCHS': '1', 'CHAL_EVAL_PERIOD': '9999', 'CHAL_SAVE_PERIOD': '9999',
        })
        rate = run('train.py', code, env)
        shutil.rmtree(scratch, ignore_errors=True)
        print('\nSpeed: %.2f iterations/s, %d iterations per epoch, %.0f minutes per epoch' % (rate, iters_per_epoch, iters_per_epoch / rate / 60))
        for epochs in (20, 30, 50, 100):
            print('  %3d epochs: about %.1f GPU-hours, without evaluation' % (epochs, epochs * iters_per_epoch / rate / 3600))
    else:
        save_dir = os.path.join(args.work, 'logs')
        resume_path = os.path.join(save_dir, 'resume.pth')
        os.makedirs(save_dir, exist_ok=True)
        restore_checkpoint(args.input, resume_path)
        env.update({
            'CHAL_TRAIN_ANNOTATION': take_lines(train_txt, os.path.join(annotations, 'run_train.txt'), args.limit_train),
            'CHAL_VAL_ANNOTATION': take_lines(val_txt, os.path.join(annotations, 'run_val.txt'), args.limit_val),
            'CHAL_SAVE_DIR': save_dir, 'CHAL_EPOCHS': str(args.epochs),
            'CHAL_EVAL_PERIOD': str(args.eval_period), 'CHAL_SAVE_PERIOD': str(args.save_period),
        })
        started = time.time()
        finished = train_until_done(code, env, resume_path, args)
        print('Finished %d of %d epochs after %.1f hours this session.' % (finished, args.epochs, (time.time() - started) / 3600))
        if finished >= args.epochs:
            teacher = os.path.join(args.work, 'teacher_ema.pth')
            torch.save(torch.load(resume_path, map_location='cpu')['ema'], teacher)
            print('Teacher saved to', teacher)
            evaluate(code, args, data_root, teacher, annotations)
        else:
            print('Not finished yet. Add this notebook version output as an input and run it again to continue.')
