"""Small dynamic loss scaler for ordinary and disk-spilled fp16 gradients.

The disk optimizer's post-accumulate hooks clear parameter.grad, so PyTorch's
GradScaler cannot inspect or unscale those gradients. This class manages only
scale/backoff/growth; the worker checks and unscales the actual optimizer data.
"""
from __future__ import annotations
import math


class DynamicLossScaler:
    MIN_SCALE = 2.0 ** -24
    def __init__(self, initial=4096.0, growth_interval=2000):
        if not math.isfinite(initial) or initial < 1 or growth_interval < 1:
            raise ValueError('Invalid fp16 dynamic loss scaling configuration.')
        self.scale = float(initial)
        self.growth_interval = int(growth_interval)
        self.good_steps = 0

    def overflow(self):
        if self.scale <= self.MIN_SCALE:
            raise FloatingPointError('FP16 gradients remain non-finite at the minimum loss scale. Training stopped without an optimizer update; review precision and learning settings.')
        self.scale = max(self.MIN_SCALE, self.scale / 2.0)
        self.good_steps = 0

    def success(self):
        self.good_steps += 1
        if self.good_steps >= self.growth_interval:
            self.scale = min(self.scale * 2.0, 16777216.0)
            self.good_steps = 0

    def state_dict(self):
        return {'scale': self.scale, 'growth_interval': self.growth_interval, 'good_steps': self.good_steps}

    def load_state_dict(self, value):
        if not isinstance(value, dict):
            raise ValueError('Checkpoint has invalid fp16 loss scaler state.')
        scale, interval, steps = value.get('scale'), value.get('growth_interval'), value.get('good_steps')
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale)
            or not self.MIN_SCALE <= scale <= 16777216.0 or type(interval) is not int or interval != self.growth_interval
            or type(steps) is not int or not 0 <= steps < interval):
            raise ValueError('Checkpoint has invalid or incompatible fp16 loss scaler state.')
        self.scale, self.good_steps = float(scale), steps
