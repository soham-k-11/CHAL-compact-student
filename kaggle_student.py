import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

import torch

from kaggle_train import find_root, finished_epochs, run, take_frames, take_lines, train_until_done

# Runs one student training on Kaggle (labels, feature, frequency, distill or robust), all three ablation runs in one
# session (ablations), or scores the teacher on background-changed clips.
# Each run has its own output folder, and a run only resumes from a checkpoint in a folder with its own name.

def find_file(input_dir, name):
    for dir_path, _, files in os.walk(input_dir, followlinks=True):
        if name in files:
            return os.path.join(dir_path, name)
    raise FileNotFoundError('No %s under %s. Attach the teacher notebook output as an input.' % (name, input_dir))

def restore_run(input_dir, tag, resume_path):
    if os.path.exists(resume_path):
        return
    best_epoch, best_path = -1, None
    for dir_path, _, files in os.walk(input_dir, followlinks=True):
        if 'resume.pth' in files and tag in dir_path:
            path = os.path.join(dir_path, 'resume.pth')
            epoch = torch.load(path, map_location='cpu')['epoch']
            if epoch > best_epoch:
                best_epoch, best_path = epoch, path
    if best_path:
        os.makedirs(os.path.dirname(resume_path), exist_ok=True)
        shutil.copy(best_path, resume_path)
        print('Continuing from %s, which finished epoch %d.' % (best_path, best_epoch + 1))

def evaluate(code, args, data_root, annotations, out, model_env, splits, threshold=None):
    results = {}
    for name, split, strength in splits:
        gt = take_frames(os.path.join(annotations, 'instances_%s.json' % split),
                         os.path.join(annotations, 'eval_%s.json' % split), args.limit_eval)
        env = dict(model_env, CHAL_COCO_GT=gt, CHAL_DATASET_IMG_PATH=data_root, CHAL_INPUT_SIZE='512', CHAL_LETTERBOX='1',
                   CHAL_BG_STRENGTH=str(strength))
        # The threshold is picked on validation and only then applied to the test set.
        if threshold is not None:
            env['CHAL_CONF_THRES'] = str(threshold)
        run('test.py', code, env)
        coco_eval = os.path.join(code, 'map_out', 'coco_eval')
        with open(os.path.join(coco_eval, 'metrics.json')) as f:
            results[name] = json.load(f)
        threshold = results[name]['threshold']

        with open(os.path.join(coco_eval, 'eval_results.json')) as f:
            detections = [d for d in json.load(f) if d['score'] >= 0.01]
        detection_path = os.path.join(out, 'detections_%s.json.gz' % name)
        with gzip.open(detection_path, 'wt') as f:
            json.dump(detections, f)
        subprocess.check_call([sys.executable, os.path.join('utils', 'det_report.py'), '--gt', gt, '--detections', detection_path,
                               '--threshold', str(threshold), '--image-root', data_root, '--out', os.path.join(out, 'report_%s.json' % name)], cwd=code)
        print('Finished scoring', name, flush=True)
    path = os.path.join(out, 'results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print('Saved to', path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['speed', 'train'], default='train')
    parser.add_argument('--run', choices=['labels', 'feature', 'frequency', 'distill', 'robust', 'teacher-bg', 'ablations'], required=True)
    parser.add_argument('--width', type=float, default=0.25)
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
    parser.add_argument('--bg-strength', type=float, default=25.0)
    parser.add_argument('--speed-iters', type=int, default=50)
    parser.add_argument('--score-hours', type=float, default=0.5, help='time set aside to score each run, in ablations')
    parser.add_argument('--limit-train', type=int, default=0, help='use only this many frames, for trial runs')
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    args = parser.parse_args()

    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or 'none')
    data_root = find_root(args.input, 'train.txt', 'test.txt', 'Sequence1')
    source = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(args.code_dir):
        shutil.copytree(source, args.code_dir, ignore=shutil.ignore_patterns('.git', 'logs*', 'datasets', '__pycache__'))
    code = args.code_dir
    annotations = os.path.join(code, 'datasets', 'NUDT-MIRSDT')
    subprocess.check_call([sys.executable, os.path.join('utils_coco', 'nudt_to_coco.py'), '--root', data_root, '--out', annotations], cwd=code)
    teacher_path = find_file(args.input, 'teacher_ema.pth')
    print('Teacher:', teacher_path)

    if args.run == 'teacher-bg':
        out = os.path.join(args.work, 'teacher_bg')
        os.makedirs(out, exist_ok=True)
        evaluate(code, args, data_root, annotations, out, {'CHAL_MODEL_PATH': teacher_path, 'CHAL_MODEL_KIND': 'teacher'},
                 [('val', 'val', 0), ('test', 'test', 0), ('test_bg', 'test', args.bg_strength)])
        sys.exit(0)

    train_txt = os.path.join(annotations, 'coco_train_NUDT.txt')
    val_txt = os.path.join(annotations, 'coco_val_NUDT.txt')
    run_train = take_lines(train_txt, os.path.join(annotations, 'run_train.txt'), args.limit_train)
    run_val = take_lines(val_txt, os.path.join(annotations, 'run_val.txt'), args.limit_val)
    with open(run_train) as f:
        iters_per_epoch = sum(1 for _ in f) // args.batch_size // 2
    env = {
        'CHAL_STUDENT_WIDTH': str(args.width), 'CHAL_TEACHER': teacher_path,
        'CHAL_BATCH_SIZE': str(args.batch_size), 'CHAL_NUM_WORKERS': str(args.num_workers),
        'CHAL_BG_STRENGTH': str(args.bg_strength),
        'CHAL_FP16': '0' if args.no_fp16 or not torch.cuda.is_available() else '1',
        'CHAL_TRAIN_ANNOTATION': run_train, 'CHAL_VAL_ANNOTATION': run_val,
    }
    student_splits = [('val', 'val', 0), ('test', 'test', 0), ('test_bg', 'test', args.bg_strength)]

    def measure_speed(run_name):
        scratch = os.path.join(code, 'speed_' + run_name)
        shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch)
        rate = run('train_student.py', code, dict(env,
            CHAL_STUDENT_MODE=run_name,
            CHAL_TRAIN_ANNOTATION=take_lines(train_txt, os.path.join(scratch, 'train.txt'), args.speed_iters * args.batch_size * 2),
            CHAL_VAL_ANNOTATION=take_lines(val_txt, os.path.join(scratch, 'val.txt'), args.batch_size * 2),
            CHAL_SAVE_DIR=os.path.join(scratch, 'logs'), CHAL_EPOCHS='1', CHAL_EVAL_PERIOD='9999', CHAL_SAVE_PERIOD='9999'))
        shutil.rmtree(scratch, ignore_errors=True)
        if not rate:
            raise RuntimeError('Could not read the training speed of %s from its progress bar.' % run_name)
        return rate

    def train_and_score(run_name, epochs, hours, eval_period, score_unfinished=False):
        tag = 'student_%s_w%03d' % (run_name, round(args.width * 100))
        out = os.path.join(args.work, tag)
        save_dir = os.path.join(out, 'logs')
        resume_path = os.path.join(save_dir, 'resume.pth')
        os.makedirs(save_dir, exist_ok=True)
        restore_run(args.input, tag, resume_path)
        run_env = dict(env, CHAL_STUDENT_MODE=run_name, CHAL_SAVE_DIR=save_dir, CHAL_EPOCHS=str(epochs),
                       CHAL_EVAL_PERIOD=str(eval_period), CHAL_SAVE_PERIOD=str(args.save_period))
        run_args = argparse.Namespace(**vars(args))
        run_args.epochs, run_args.max_hours = epochs, hours
        finished = train_until_done(code, run_env, resume_path, run_args, 'train_student.py')
        print('%s: finished %d of %d epochs.' % (run_name, finished, epochs), flush=True)
        if finished == 0 or (finished < epochs and not score_unfinished):
            print('Not finished yet. Add this notebook version output as an input and run it again to continue.')
            return None
        weights = os.path.join(out, 'student_ema.pth')
        torch.save(torch.load(resume_path, map_location='cpu')['ema'], weights)
        print('Student saved to', weights)
        evaluate(code, args, data_root, annotations, out,
                 {'CHAL_MODEL_PATH': weights, 'CHAL_MODEL_KIND': 'student', 'CHAL_STUDENT_WIDTH': str(args.width)}, student_splits)
        with open(os.path.join(out, 'results.json')) as f:
            results = json.load(f)
        results['finished_epochs'], results['planned_epochs'] = finished, epochs
        with open(os.path.join(out, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        return out

    if args.mode == 'speed':
        rate = measure_speed(args.run)
        print('\nSpeed: %.2f iterations/s, %d iterations per epoch, %.0f minutes per epoch' % (rate, iters_per_epoch, iters_per_epoch / rate / 60))
        print('%d epochs: about %.1f GPU-hours, without evaluation' % (args.epochs, args.epochs * iters_per_epoch / rate / 3600))
        sys.exit(0)

    if args.run != 'ablations':
        train_and_score(args.run, args.epochs, args.max_hours, args.eval_period)
        sys.exit(0)

    # All three student ablations in one session: the same number of epochs for each, chosen from measured speed
    # so that training and scoring of all three fit in --max-hours. A run that fails does not stop the others.
    started = time.time()
    runs = ['labels', 'feature', 'frequency']
    rate = {'labels': measure_speed('labels'), 'frequency': measure_speed('frequency')}
    rate['feature'] = rate['frequency']
    # Hours per epoch: training at the measured rate plus a quarter for validation and slow batches.
    epoch_hours = {r: iters_per_epoch / rate[r] / 3600 * 1.25 for r in runs}
    score_hours = args.score_hours
    left = args.max_hours - (time.time() - started) / 3600 - len(runs) * score_hours - 0.25
    epochs = min(args.epochs, max(3, int(left / sum(epoch_hours.values()))))
    print('\nSpeed: labels %.2f it/s, with teacher %.2f it/s. Plan: %d epochs for each of %s, about %.1f hours of training.' % (
        rate['labels'], rate['frequency'], epochs, ', '.join(runs), epochs * sum(epoch_hours.values())), flush=True)

    summary = {'epochs': epochs, 'iters_per_epoch': iters_per_epoch, 'speed_it_per_s': rate, 'runs': {}}
    for i, run_name in enumerate(runs):
        # Each run may use what is left after setting aside the planned time of the runs still to come.
        later = sum(epochs * epoch_hours[r] + score_hours for r in runs[i + 1:])
        hours = args.max_hours - (time.time() - started) / 3600 - later - score_hours
        print('\n===== %s: %d epochs, up to %.1f hours of training =====' % (run_name, epochs, hours), flush=True)
        try:
            out = train_and_score(run_name, epochs, max(hours, 0.05), 9999, score_unfinished=True)
            if out:
                with open(os.path.join(out, 'results.json')) as f:
                    summary['runs'][run_name] = json.load(f)
                with open(os.path.join(out, 'report_test.json')) as f:
                    report = json.load(f)
                summary['runs'][run_name]['test_low_contrast'] = report['low_contrast']
                summary['runs'][run_name]['test_other'] = report['other']
        except Exception as error:
            print('%s failed: %s. Going on with the next run.' % (run_name, error), flush=True)
            summary['runs'][run_name] = {'error': str(error)}
        with open(os.path.join(args.work, 'ablation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print('\n===== Summary, test set =====')
    print('%-10s %7s %6s %6s %7s %7s %10s' % ('run', 'epochs', 'Pd', 'F1', 'FA/fr', 'mAP50', 'Pd low-c'))
    for run_name, result in summary['runs'].items():
        if 'test' in result:
            t = result['test']
            low = result.get('test_low_contrast') or {}
            print('%-10s %7s %6.3f %6.3f %7.3f %7.3f %10.3f' % (run_name, '%d/%d' % (result['finished_epochs'], result['planned_epochs']),
                  t['pd'], t['f1'], t['false_alarms_per_frame'], t['map50'], low.get('pd', float('nan'))))
        else:
            print('%-10s failed: %s' % (run_name, result.get('error')))
    print('Saved to', os.path.join(args.work, 'ablation_summary.json'))
