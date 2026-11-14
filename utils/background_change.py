import numpy as np
import torch
import torch.nn.functional as F

# Label-preserving background change: a smooth brightness pattern added to a whole clip, faded out around the target.
# The same pattern is used for every frame, so it adds no false motion. Only the newest frame's box is known here,
# so the margin also has to cover the target's recent movement.

GREY_STD = float(np.mean([0.229, 0.224, 0.225]))

def smooth_field(batch, size, cells, generator, device):
    field = torch.randn(batch, 1, cells, cells, generator=generator).to(device)
    field = F.interpolate(field, size=size, mode='bicubic', align_corners=False)
    return field / field.abs().flatten(1).max(1).values.view(-1, 1, 1, 1).clamp(min=1e-6)

def target_keep_out(boxes, size, margin, device):
    # boxes: for each clip, an array of centre x, centre y, width, height in input pixels.
    mask = torch.ones(len(boxes), 1, size[0], size[1], device=device)
    for i, clip_boxes in enumerate(boxes):
        for box in clip_boxes:
            cx, cy, w, h = [float(v) for v in box[:4]]
            x0, x1 = max(int(cx - w / 2 - margin), 0), int(cx + w / 2 + margin) + 1
            y0, y1 = max(int(cy - h / 2 - margin), 0), int(cy + h / 2 + margin) + 1
            mask[i, :, y0:y1, x0:x1] = 0
    # Soften the edge, and keep the cleared area fully protected.
    return torch.minimum(mask, F.avg_pool2d(mask, 2 * margin + 1, 1, margin)) if margin > 0 else mask

def background_change(images, boxes, strength=25.0, margin=12, generator=None):
    # images: batch, channels, frames, height, width, normalised. strength: peak change in grey levels out of 255.
    size = images.shape[-2:]
    device = images.device
    pattern = smooth_field(len(images), size, 4, generator, device) + 0.5 * smooth_field(len(images), size, 16, generator, device)
    pattern = pattern * target_keep_out(boxes, size, margin, device) * strength / 1.5 / 255.0 / GREY_STD
    return images + pattern.unsqueeze(2)
