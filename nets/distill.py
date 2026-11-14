import torch
import torch.nn as nn
import torch.nn.functional as F

# Teacher responses and the losses that make the student match them. CHAL's code is untouched: its internal
# responses are read with hooks.

class TeacherResponses:
    def __init__(self, teacher):
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.captured = {}
        teacher.neck.stst_processor.register_forward_hook(self.keep('anomaly'))
        teacher.neck.feature_modulation.register_forward_hook(self.keep('feature'))

    def keep(self, name):
        def hook(module, inputs, output):
            self.captured[name] = output
        return hook

    @torch.no_grad()
    def __call__(self, images):
        prediction = self.teacher(images)[0]
        return {'prediction': prediction, 'feature': self.captured['feature'], 'anomaly': self.captured['anomaly'].clamp(0, 1)}

def teacher_weight(prediction, foreground_weight):
    # Almost every grid cell is background, so cells the teacher believes in count for much more.
    confidence = torch.sigmoid(prediction[:, 4:5]) * torch.sigmoid(prediction[:, 5:]).max(1, keepdim=True).values
    return confidence, 1.0 + foreground_weight * confidence

def weighted_mean(values, weight):
    return (values * weight).sum() / weight.sum().clamp(min=1e-6)

class DistillLoss(nn.Module):
    def __init__(self, student_dim, teacher_dim=128, foreground_weight=50.0):
        super().__init__()
        self.foreground_weight = foreground_weight
        # Only used in training, so it is left out of the student's size and compute.
        self.adapter = nn.Conv2d(student_dim, teacher_dim, 1)

    def prediction_loss(self, student, target, confidence, weight):
        # Heads: box 0-3, objectness 4, classes 5 onward.
        box = F.smooth_l1_loss(student[:, :4], target[:, :4], reduction='none').mean(1, keepdim=True)
        objectness = F.binary_cross_entropy_with_logits(student[:, 4:5], torch.sigmoid(target[:, 4:5]), reduction='none')
        classes = F.binary_cross_entropy_with_logits(student[:, 5:], torch.sigmoid(target[:, 5:]), reduction='none').mean(1, keepdim=True)
        return weighted_mean(box, confidence.clamp(min=1e-6)) + weighted_mean(objectness + classes, weight)

    def forward(self, student, teacher):
        confidence, weight = teacher_weight(teacher['prediction'].detach(), self.foreground_weight)
        target = teacher['feature'].detach()
        target = (target - target.mean((2, 3), keepdim=True)) / (target.std((2, 3), keepdim=True) + 1e-6)
        losses = {
            'prediction': self.prediction_loss(student['prediction'], teacher['prediction'].detach(), confidence, weight),
            'anomaly': weighted_mean((student['anomaly'] - teacher['anomaly'].detach()) ** 2, weight),
            'feature': weighted_mean(((self.adapter(student['feature']) - target) ** 2).mean(1, keepdim=True), weight),
        }
        return losses

    def normalised_pair(self, student, teacher):
        target = teacher['feature'].detach().float()
        target = (target - target.mean((2, 3), keepdim=True)) / (target.std((2, 3), keepdim=True) + 1e-6)
        return self.adapter(student['feature']).float(), target

    def feature_loss(self, student, teacher):
        # Conventional feature distillation: a plain squared error over the whole map.
        output, target = self.normalised_pair(student, teacher)
        return ((output - target) ** 2).mean()

    def frequency_loss(self, student, teacher, cutoff=0.25, high_weight=2.0, rule='balanced', bins=8):
        # Frequency-aligned distillation: the same error, measured on the 2-D spectrum and weighted by band. With an
        # orthonormal transform and equal weights it equals feature_loss exactly, so the two differ only in weighting.
        #   balanced  each of `bins` radial bands is weighted by the inverse of the teacher's power in it (capped at 10x), so every
        #             band counts equally instead of the strong low frequencies dominating. It assumes nothing about
        #             where targets sit and has no weight to tune.
        #   highpass  bands above `cutoff` (a fraction of the highest frequency) weighted by high_weight.
        output, target = self.normalised_pair(student, teacher)
        h, w = output.shape[-2:]
        fy = torch.fft.fftfreq(h, device=output.device).abs().view(-1, 1)
        fx = torch.fft.rfftfreq(w, device=output.device).view(1, -1)
        radius = torch.sqrt(fy ** 2 + fx ** 2) / 0.5
        # rfft2 keeps half the spectrum; inner columns stand for two frequencies, so they count twice.
        count = torch.full((1, w // 2 + 1), 2.0, device=output.device)
        count[0, 0] = 1.0
        if w % 2 == 0:
            count[0, -1] = 1.0
        target_spectrum = torch.fft.rfft2(target, norm='ortho')
        difference = torch.fft.rfft2(output, norm='ortho') - target_spectrum
        if rule == 'highpass':
            weight = 1.0 + (high_weight - 1.0) * (radius > cutoff).float()
        else:
            band = torch.clamp((radius / radius.max() * bins).long(), max=bins - 1)
            cells = (count.expand_as(radius) * torch.ones_like(radius)).flatten()
            power = (target_spectrum.abs() ** 2).mean((0, 1)).detach() * count
            band_power = torch.zeros(bins, device=output.device).index_add_(0, band.flatten(), power.flatten())
            band_cells = torch.zeros(bins, device=output.device).index_add_(0, band.flatten(), cells)
            band_weight = 1.0 / (band_power / band_cells.clamp(min=1) + 1e-6)
            # A band where the teacher has almost no power is mostly noise, so no band may weigh over 10x the strongest.
            band_weight = torch.minimum(band_weight, 10.0 * band_weight.min())
            band_weight = band_weight / (band_weight * band_cells).sum() * band_cells.sum()
            weight = band_weight[band]
        return (difference.abs() ** 2 * count * weight).sum() / output.numel()

    def invariance_loss(self, changed, clean):
        # The same scene with a different background should give the same answer.
        _, weight = teacher_weight(clean['prediction'].detach(), self.foreground_weight)
        objectness = F.binary_cross_entropy_with_logits(changed['prediction'][:, 4:5], torch.sigmoid(clean['prediction'][:, 4:5].detach()), reduction='none')
        anomaly = (changed['anomaly'] - clean['anomaly'].detach()) ** 2
        return weighted_mean(objectness + anomaly, weight)
