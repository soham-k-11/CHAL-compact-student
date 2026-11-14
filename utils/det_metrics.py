from collections import defaultdict

import numpy as np

# Frame-level detection metrics at one confidence threshold. A detection matches a target when their
# centres are within max_distance pixels of the original image, the usual small-target tolerance.

def _centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2

def operating_point(frames, annotations, detections, threshold, max_distance=3.0):
    targets_by_frame = defaultdict(list)
    for ann in annotations:
        targets_by_frame[ann['image_id']].append(_centre(ann['bbox']))

    dets_by_frame = defaultdict(list)
    for det in detections:
        if det['score'] >= threshold:
            dets_by_frame[det['image_id']].append(det)

    num_targets = matched = false_alarms = false_pixels = total_pixels = 0
    for frame in frames:
        total_pixels += frame['width'] * frame['height']
        targets = list(targets_by_frame[frame['id']])
        num_targets += len(targets)

        for det in sorted(dets_by_frame[frame['id']], key=lambda d: d['score'], reverse=True):
            cx, cy = _centre(det['bbox'])
            distances = [np.hypot(cx - tx, cy - ty) for tx, ty in targets]
            if distances and min(distances) <= max_distance:
                targets.pop(int(np.argmin(distances)))
                matched += 1
            else:
                false_alarms += 1
                false_pixels += max(det['bbox'][2], 0) * max(det['bbox'][3], 0)

    pd = matched / max(num_targets, 1)
    precision = matched / max(matched + false_alarms, 1)
    f1 = 2 * pd * precision / max(pd + precision, 1e-12)
    return {
        'threshold': float(threshold),
        'pd': pd,
        'precision': precision,
        'f1': f1,
        'false_alarms': false_alarms,
        'false_alarms_per_frame': false_alarms / max(len(frames), 1),
        'fa_pixel': false_pixels / max(total_pixels, 1),
        'num_targets': num_targets,
        'num_frames': len(frames),
    }

def best_operating_point(frames, annotations, detections, thresholds=np.arange(0.05, 1.0, 0.05)):
    points = [operating_point(frames, annotations, detections, t) for t in thresholds]
    return max(points, key=lambda p: p['f1'])
