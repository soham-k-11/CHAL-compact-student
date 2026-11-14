import math

import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer
from .slowfastnet import YOLOXHead

# Compact student for CHAL. Each frame goes through the backbone once and only a small projection of it is kept,
# so a new frame costs one backbone pass. The background is the temporal median of the kept projections, replacing
# the teacher's neural background field and Swin transformer. Only past frames are used, so it can run live.

class CompactStudent(nn.Module):
    def __init__(self, num_classes, num_frame=5, width=0.25, history_dim=16):
        super().__init__()
        self.num_frame = num_frame
        self.history_dim = history_dim
        c3, c4, c5 = int(256 * width), int(512 * width), int(1024 * width)
        self.feature_dim = c3

        self.backbone = CSPDarknet(0.33, width)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce5 = BaseConv(c5, c4, 1, 1)
        self.fuse4 = CSPLayer(2 * c4, c4, 1, False)
        self.reduce4 = BaseConv(c4, c3, 1, 1)
        self.fuse3 = CSPLayer(2 * c3, c3, 1, False)

        self.project = nn.Conv2d(c3, history_dim, 1)
        self.anomaly = nn.Sequential(
            BaseConv(history_dim, history_dim, 3, 1),
            nn.Conv2d(history_dim, 1, 1),
            nn.Sigmoid()
        )
        self.fuse = BaseConv(c3 + history_dim, c3, 3, 1)
        self.head = YOLOXHead(num_classes, width=width, in_channels=[256])

    def init_prior(self, probability=0.01):
        # Start by predicting "almost certainly no target", as nearly every grid cell is empty. Without pretrained
        # weights the student would otherwise spend its early training unlearning a 50% guess.
        bias = -math.log((1 - probability) / probability)
        for layer in list(self.head.obj_preds) + list(self.head.cls_preds):
            nn.init.constant_(layer.bias, bias)

    def encode(self, frame):
        out = self.backbone(frame)
        x = self.fuse4(torch.cat([self.upsample(self.reduce5(out["dark5"])), out["dark4"]], 1))
        feature = self.fuse3(torch.cat([self.upsample(self.reduce4(x)), out["dark3"]], 1))
        return feature, self.project(feature)

    def decode(self, feature, projection, history):
        # history: batch, past frames, history_dim, height, width
        background = history.median(dim=1).values
        residual = torch.abs(projection - background)
        anomaly = self.anomaly(residual)
        fused = self.fuse(torch.cat([feature * (1.0 + anomaly), residual], 1))
        return self.head([fused]), fused, anomaly

    def forward(self, inputs, return_responses=False):
        b, c, t, h, w = inputs.shape
        feature, projection = self.encode(inputs.transpose(1, 2).reshape(b * t, c, h, w))
        current = feature.view(b, t, *feature.shape[1:])[:, -1]
        projection = projection.view(b, t, *projection.shape[1:])
        outputs, fused, anomaly = self.decode(current, projection[:, -1], projection[:, :-1])
        if return_responses:
            return {'prediction': outputs[0], 'feature': fused, 'anomaly': anomaly}
        return outputs

class StreamingStudent:
    # Frame-by-frame use of a CompactStudent. The first frame fills the buffer, as the dataset repeats it at sequence start.
    def __init__(self, model):
        self.model = model.eval()
        self.history = None

    def reset(self):
        self.history = None

    def buffer_bytes(self):
        return 0 if self.history is None else self.history.numel() * self.history.element_size()

    @torch.no_grad()
    def __call__(self, frame):
        feature, projection = self.model.encode(frame)
        if self.history is None:
            self.history = projection.unsqueeze(1).repeat(1, self.model.num_frame - 1, 1, 1, 1)
        outputs, _, _ = self.model.decode(feature, projection, self.history)
        self.history = torch.cat([self.history[:, 1:], projection.unsqueeze(1)], 1)
        return outputs
