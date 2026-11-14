import argparse
import collections
import gzip
import json
import os

import numpy as np
from PIL import Image

if __package__:
    from .det_metrics import operating_point
else:
    from det_metrics import operating_point

# Breaks a detection result down by sequence and by how much the targets stand out from their surroundings.
# Contrast is (brightest target pixel - local background mean) / local background spread, the median over a sequence.

LOW_CONTRAST = 2.0
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.75, 0.9)

def sequence_of(image):
    return image['file_name'].split('/')[0]

def target_contrast(image_root, gt, every=5, margin=6):
    images = {image['id']: image for image in gt['images']}
    values = collections.defaultdict(list)
    for ann in gt['annotations'][::every]:
        image = images[ann['image_id']]
        pixels = np.array(Image.open(os.path.join(image_root, image['file_name'])).convert('L'), dtype=float)
        x, y, w, h = [int(round(v)) for v in ann['bbox']]
        y0, x0 = max(0, y - margin), max(0, x - margin)
        patch = pixels[y0:y + h + margin, x0:x + w + margin]
        keep = np.ones_like(patch, dtype=bool)
        keep[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
        target = pixels[y:y + h, x:x + w]
        if target.size and keep.any():
            values[sequence_of(image)].append((target.max() - patch[keep].mean()) / (patch[keep].std() + 1e-6))
    return {seq: float(np.median(v)) for seq, v in values.items()}

def load_detections(path):
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        return json.load(f)

def report(gt, detections, threshold, contrast, max_distance=3.0):
    frames_of = collections.defaultdict(list)
    for image in gt['images']:
        frames_of[sequence_of(image)].append(image)

    def point(sequences, value=threshold):
        frames = [f for seq in sequences for f in frames_of[seq]]
        ids = {f['id'] for f in frames}
        return operating_point(frames, [a for a in gt['annotations'] if a['image_id'] in ids],
                               [d for d in detections if d['image_id'] in ids], value, max_distance)

    low = [seq for seq in frames_of if contrast.get(seq, np.inf) < LOW_CONTRAST]
    other = [seq for seq in frames_of if seq not in low]
    return {
        'threshold': threshold,
        'all': point(list(frames_of)),
        'low_contrast': point(low) if low else None,
        'other': point(other) if other else None,
        'low_contrast_sequences': sorted(low),
        'sweep': [point(list(frames_of), t) for t in THRESHOLDS],
        'per_sequence': {seq: dict(point([seq]), contrast=contrast.get(seq)) for seq in sorted(frames_of)},
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    with open(args.gt) as f:
        ground_truth = json.load(f)
    result = report(ground_truth, load_detections(args.detections), args.threshold, target_contrast(args.image_root, ground_truth))
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    for name in ('all', 'low_contrast', 'other'):
        p = result[name]
        if p:
            print('%-13s Pd %.3f  precision %.3f  F1 %.3f  false alarms/frame %.3f  (%d targets)' % (name, p['pd'], p['precision'], p['f1'], p['false_alarms_per_frame'], p['num_targets']))
