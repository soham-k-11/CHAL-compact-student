import argparse
import copy
import resource
import time

import torch
from thop import profile

from nets.slowfastnet import slowfastnet
from nets.student import CompactStudent, StreamingStudent

# Size, compute, latency and memory of the teacher and a student on this machine's CPU. thop only counts layers with
# weights, so the student's median and the teacher's non-layer operations are not in the MAC figures.

class PerFrame(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net.backbone(x)
        z = self.net.conv1(torch.cat([self.net.mulit1(y[2]), y[1]], 1))
        return self.net.conv2(torch.cat([self.net.mulit2(z), y[0]], 1)), self.net.conv1(torch.cat([self.net.mulit1(y[2]), y[1]], 1))

class StudentFrame(torch.nn.Module):
    def __init__(self, net, history):
        super().__init__()
        self.net = net
        self.history = history

    def forward(self, x):
        feature, projection = self.net.encode(x)
        return self.net.decode(feature, projection, self.history)[0]

def energy_uj():
    # The whole processor package, which needs read permission (usually root).
    try:
        with open('/sys/class/powercap/intel-rapl:0/energy_uj') as f:
            return int(f.read())
    except OSError:
        return None

def latency(call, threads, runs=30):
    torch.set_num_threads(threads)
    for _ in range(3):
        call()
    times = []
    energy_start = energy_uj()
    started = time.perf_counter()
    for _ in range(runs):
        t = time.perf_counter()
        call()
        times.append(time.perf_counter() - t)
    elapsed = time.perf_counter() - started
    energy_end = energy_uj()
    times = sorted(times)
    joules = None if energy_start is None or energy_end is None else (energy_end - energy_start) / 1e6 / runs
    return {'median_ms': times[len(times) // 2] * 1e3, 'p95_ms': times[int(len(times) * 0.95)] * 1e3, 'joules_per_call': joules, 'seconds': elapsed}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['teacher', 'student'], required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--width', type=float, default=0.25)
    parser.add_argument('--threads', default='8,1')
    parser.add_argument('--runs', type=int, default=30)
    args = parser.parse_args()
    size = 512
    frame = torch.rand(1, 3, size, size)
    clip = torch.rand(1, 3, 5, size, size)

    if args.kind == 'teacher':
        model = slowfastnet(1, num_frame=5)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        model.eval()
        total, _ = profile(copy.deepcopy(model), inputs=(clip,), verbose=False)
        one, _ = profile(PerFrame(copy.deepcopy(model)), inputs=(frame,), verbose=False)
        print('Parameters %.3f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
        print('MACs per clip, as written: %.2f G' % (total / 1e9))
        print('MACs per new frame if it also kept past backbone features: %.2f G' % ((total - 4 * one) / 1e9))
        print('Buffer: 4 past 8-bit grey frames %.2f MB' % (4 * size * size / 1e6))
        with torch.no_grad():
            for threads in [int(t) for t in args.threads.split(',')]:
                print('Latency per new frame, %d threads:' % threads, latency(lambda: model(clip), threads, args.runs))
    else:
        model = CompactStudent(1, num_frame=5, width=args.width)
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        model.eval()
        stream = StreamingStudent(model)
        stream(frame)
        macs, _ = profile(StudentFrame(copy.deepcopy(model), stream.history.clone()), inputs=(frame,), verbose=False)
        print('Parameters %.3f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
        print('MACs per new frame: %.2f G' % (macs / 1e9))
        print('Buffer: 4 past projections, 32-bit %.2f MB' % (stream.buffer_bytes() / 1e6))
        for threads in [int(t) for t in args.threads.split(',')]:
            print('Latency per new frame, %d threads:' % threads, latency(lambda: stream(frame), threads, args.runs))
    print('Peak process memory %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    print('Energy counter:', 'available' if energy_uj() is not None else 'not readable on this machine')
