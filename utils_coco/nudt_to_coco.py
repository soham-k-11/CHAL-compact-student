import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

# Converts NUDT-MIRSDT (per-frame masks, train.txt/test.txt) into the box annotations train.py and test.py expect.
# Validation sequences are held out of the train split, so the test split is never used to pick a model.

def read_split(path):
    frames = []
    with open(path) as f:
        for line in f.read().split():
            parts = line.split('/')
            frames.append((parts[0], os.path.splitext(parts[-1])[0]))
    return frames

def mask_to_boxes(mask_path):
    mask = np.array(Image.open(mask_path)) > 0
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    boxes = []
    for obj in ndimage.find_objects(labels):
        boxes.append((obj[1].start, obj[0].start, obj[1].stop, obj[0].stop))
    return boxes, mask.shape

def write_txt(path, root, frames, boxes):
    with open(path, 'w') as f:
        for seq, name in frames:
            f.write(os.path.join(root, seq, 'images', name + '.png'))
            for box in boxes[(seq, name)][0]:
                f.write(" %d,%d,%d,%d,0" % box)
            f.write('\n')

def write_coco(path, frames, boxes):
    images, annotations = [], []
    for image_id, (seq, name) in enumerate(frames, 1):
        frame_boxes, (h, w) = boxes[(seq, name)]
        images.append({'id': image_id, 'file_name': os.path.join(seq, 'images', name + '.png'), 'width': w, 'height': h})
        for x1, y1, x2, y2 in frame_boxes:
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id, 'category_id': 1,
                                'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
    with open(path, 'w') as f:
        json.dump({'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'target'}]}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='folder holding Sequence*/, train.txt and test.txt')
    parser.add_argument('--out', default='datasets/NUDT-MIRSDT')
    parser.add_argument('--val-seqs', type=int, default=8, help='train sequences held out for validation')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--val-list', default='', help='comma-separated validation sequences; overrides --val-seqs')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)

    train = read_split(os.path.join(root, 'train.txt'))
    test = read_split(os.path.join(root, 'test.txt'))

    train_seqs = sorted({seq for seq, _ in train}, key=lambda s: int(s[len('Sequence'):]))
    if args.val_list:
        val_seqs = set(args.val_list.split(','))
        assert val_seqs <= set(train_seqs), 'validation sequences must come from train.txt: %s' % sorted(val_seqs - set(train_seqs))
    else:
        val_seqs = set(np.random.RandomState(args.seed).choice(train_seqs, args.val_seqs, replace=False).tolist())
    val = [frame for frame in train if frame[0] in val_seqs]
    train = [frame for frame in train if frame[0] not in val_seqs]

    boxes = {}
    for seq, name in train + val + test:
        boxes[(seq, name)] = mask_to_boxes(os.path.join(root, seq, 'masks', name + '.png'))

    write_txt(os.path.join(args.out, 'coco_train_NUDT.txt'), root, train, boxes)
    write_txt(os.path.join(args.out, 'coco_val_NUDT.txt'), root, val, boxes)
    write_txt(os.path.join(args.out, 'coco_test_NUDT.txt'), root, test, boxes)
    write_coco(os.path.join(args.out, 'instances_val.json'), val, boxes)
    write_coco(os.path.join(args.out, 'instances_test.json'), test, boxes)

    # Every sequence in the folder is accounted for: the ones in neither official list are recorded as unused.
    order = lambda names: sorted(names, key=lambda s: int(s[len('Sequence'):]))
    listed = {seq for seq, _ in train + val + test}
    unused = [d for d in os.listdir(root) if d.startswith('Sequence') and d not in listed]
    with open(os.path.join(args.out, 'split_manifest.json'), 'w') as f:
        json.dump({'train': order({s for s, _ in train}), 'val': order(val_seqs), 'test': order({s for s, _ in test}),
                   'unused': order(unused)}, f, indent=1)

    for split, frames in (('train', train), ('val', val), ('test', test)):
        print('%s: %d frames, %d sequences, %d without targets' % (split, len(frames), len({s for s, _ in frames}),
                                                                   sum(not boxes[f][0] for f in frames)))
    print('val sequences:', sorted(val_seqs, key=lambda s: int(s[len('Sequence'):])))
