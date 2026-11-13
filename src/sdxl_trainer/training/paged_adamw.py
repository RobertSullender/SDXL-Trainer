"""Optional bitsandbytes paged AdamW: explicit runtime checks and resumable state.

This is an independently implemented integration. It deliberately does not
provide fp32 master weights for reduced-precision model parameters.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys


SUPPORTED_BITSANDBYTES = {'0.49.2'}


def prepare_paged_gradients(parameters, loss_scale, max_grad_norm, chunk_elements=4 * 1024 * 1024):
    """Compute a safe global norm, then unscale/clip in the live gradient dtype."""
    import math
    import torch
    from .checkpoint_stream import _chunks
    from .disk_optimizer import NonFiniteGradients
    if not math.isfinite(loss_scale) or loss_scale <= 0 or max_grad_norm < 0 or chunk_elements < 1:
        raise ValueError('Invalid paged optimizer loss scale or gradient limit.')
    squared = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        for chunk in _chunks(gradient.detach(), chunk_elements):
            # FP64 accumulation avoids fp16/bf16 norm overflow and remains
            # bounded; only a scalar leaves the GPU at the end of the pass.
            norm = torch.linalg.vector_norm(chunk, dtype=torch.float64)
            squared = norm.square() if squared is None else squared + norm.square()
    norm = (squared.sqrt().item() / loss_scale) if squared is not None else 0.0
    if not math.isfinite(norm):
        raise NonFiniteGradients('Non-finite gradients; no optimizer update was applied.')
    factor = (min(1.0, max_grad_norm / (norm + 1e-6)) if max_grad_norm > 0 else 1.0) / loss_scale
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                # bitsandbytes selects kernels from the gradient dtype and
                # expects it to match the live parameter. Direct 16-bit
                # unscale/clip can round or underflow; the UI discloses this.
                parameter.grad.mul_(factor)
    return norm


def check_paged_finite(optimizer, parameters, chunk_elements=4 * 1024 * 1024):
    """Reject non-finite weights or floating optimizer state without large copies."""
    import torch
    from .checkpoint_stream import _chunks
    checks = {}
    def check(tensor, nonnegative=False):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return
        for chunk in _chunks(tensor.detach(), chunk_elements):
            valid = torch.isfinite(chunk).all()
            if nonnegative:
                valid = valid & (chunk >= 0).all()
            device = str(chunk.device)
            checks[device] = valid if device not in checks else checks[device] & valid
    for parameter in parameters:
        check(parameter)
        for name, value in optimizer.state.get(parameter, {}).items():
            check(value, nonnegative=name.startswith('absmax'))
    if any(not value.item() for value in checks.values()):
        raise FloatingPointError('Paged 8-bit AdamW produced non-finite weights or invalid optimizer state; no checkpoint was overwritten.')


def load_paged_runtime(cuda_override=''):
    """Select an explicit CUDA binary and reject deferred native-load errors."""
    if cuda_override and (not cuda_override.isascii() or not cuda_override.isdigit() or len(cuda_override) not in (3, 4)):
        raise ValueError('bitsandbytes CUDA override must be empty or a CUDA binary version such as 130.')
    existing = os.environ.get('BNB_CUDA_VERSION', '')
    if 'bitsandbytes' in sys.modules and existing != cuda_override:
        raise RuntimeError('bitsandbytes was already imported with a different CUDA override; restart this isolated worker.')
    if cuda_override:
        os.environ['BNB_CUDA_VERSION'] = cuda_override
    else:
        os.environ.pop('BNB_CUDA_VERSION', None)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Paged 8-bit AdamW requires an available CUDA GPU.')
        bnb = importlib.import_module('bitsandbytes')
        native = importlib.import_module('bitsandbytes.cextension').lib
        version = str(bnb.__version__)
        if version not in SUPPORTED_BITSANDBYTES:
            raise RuntimeError(f'bitsandbytes {version} is not yet validated for paged checkpoint restoration; use 0.49.2 or Disk AdamW.')
        if not getattr(native, 'compiled_with_cuda', False):
            raise RuntimeError('The selected bitsandbytes CUDA native library did not load. Check the installed binary and explicit CUDA override.')
        getattr(bnb.optim, 'PagedAdamW8bit')
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError('Paged 8-bit AdamW requires a working local bitsandbytes CUDA runtime; use Disk AdamW or repair the runtime.') from exc
    info = {'bitsandbytes': version, 'torch': str(torch.__version__), 'torch_cuda': str(torch.version.cuda),
            'cuda_override': cuda_override, 'device_capability': list(torch.cuda.get_device_capability())}
    return bnb, info


def probe_paged_runtime(bnb, precision):
    """Exercise native update and managed allocation before loading an SDXL model."""
    import torch
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[precision]
    try:
        parameter = torch.nn.Parameter(torch.ones(100_352, device='cuda', dtype=dtype))
        optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        torch.cuda.synchronize()
        state = optimizer.state[parameter]
        if not getattr(state.get('state1'), 'is_paged', False) or state['state1'].dtype != torch.uint8:
            raise RuntimeError('The optimizer did not create the expected paged 8-bit state.')
        if not torch.isfinite(parameter).all().item() or parameter[0].item() == 1.0:
            raise RuntimeError('The optimizer native probe did not produce a finite weight update.')
    except Exception as exc:
        raise RuntimeError('Paged 8-bit AdamW native CUDA test failed. Select a compatible local CUDA binary or use Disk AdamW.') from exc


def validate_paged_snapshot(source, runtime):
    """Check snapshot identity and tensor header without mutating the run."""
    from safetensors import safe_open
    source = Path(source)
    if source.is_symlink() or any((source / name).is_symlink() or not (source / name).is_file()
                                  for name in ('paged.json', 'state.safetensors')):
        raise ValueError('Paged optimizer checkpoint must contain regular state files.')
    metadata = json.loads((source / 'paged.json').read_text())
    if not isinstance(metadata, dict):
        raise ValueError('Paged optimizer checkpoint metadata must be an object.')
    if metadata.get('version') != 1 or metadata.get('runtime') != runtime:
        raise ValueError('Paged optimizer checkpoint runtime differs; resume with the original bitsandbytes, PyTorch, CUDA binary and GPU capability.')
    records, groups = metadata.get('records'), metadata.get('groups')
    if not isinstance(records, list) or not isinstance(groups, list):
        raise ValueError('Paged optimizer checkpoint parameter groups differ.')
    expected = set()
    for record in records:
        if not isinstance(record, dict) or type(record.get('group')) is not int or type(record.get('index')) is not int:
            raise ValueError('Paged optimizer checkpoint parameter layout is invalid.')
        keys = record.get('tensor_keys')
        if not isinstance(keys, list) or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('Paged optimizer checkpoint tensor metadata is invalid.')
        expected.update(f"{record['group']}.{record['index']}.{key}" for key in keys)
    with safe_open(str(source / 'state.safetensors'), framework='pt', device='cpu') as saved:
        if set(saved.keys()) != expected:
            raise ValueError('Paged optimizer checkpoint tensor names differ.')
    return metadata


class PagedAdamWCheckpoint:
    """Snapshot 8-bit states and recreate their original paged allocations."""
    def __init__(self, optimizer, named_groups, runtime):
        self.optimizer = optimizer
        self.named_groups = named_groups
        self.runtime = runtime

    def _parameters(self):
        for group_index, group in enumerate(self.named_groups):
            for parameter_index, (name, parameter) in enumerate(group['named_params']):
                yield group_index, parameter_index, name, parameter

    def save_state(self, destination):
        from .checkpoint_stream import save_safetensors_stream
        destination = Path(destination)
        if destination.is_symlink():
            raise ValueError('Paged optimizer checkpoint must use a real directory.')
        destination.mkdir(parents=True, exist_ok=True)
        check_paged_finite(self.optimizer, [p for _, _, _, p in self._parameters()])
        tensors = {}
        records = []
        for group, index, name, parameter in self._parameters():
            state = self.optimizer.state.get(parameter, {})
            scalar = {}
            keys = []
            for key, value in sorted(state.items()):
                if hasattr(value, 'dtype') and hasattr(value, 'shape'):
                    tensor_name = f'{group}.{index}.{key}'
                    tensors[tensor_name] = value
                    keys.append(key)
                elif isinstance(value, (int, float, bool)):
                    scalar[key] = value
                else:
                    raise ValueError(f'Unsupported paged optimizer state value: {key}.')
            records.append({'group': group, 'index': index, 'name': name, 'shape': list(parameter.shape),
                            'dtype': str(parameter.dtype), 'scalar': scalar, 'tensor_keys': keys})
        save_safetensors_stream(tensors, destination / 'state.safetensors')
        groups = [{key: value for key, value in group.items() if key != 'params'} for group in self.optimizer.param_groups]
        with (destination / 'paged.json').open('x', encoding='utf-8') as stream:
            json.dump({'version': 1, 'runtime': self.runtime, 'groups': groups, 'records': records}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def load_state(self, source):
        from safetensors import safe_open
        source = Path(source)
        metadata = validate_paged_snapshot(source, self.runtime)
        parameters = list(self._parameters())
        records = metadata.get('records')
        groups = metadata.get('groups')
        if not isinstance(records, list) or len(records) != len(parameters) or not isinstance(groups, list) or len(groups) != len(self.optimizer.param_groups):
            raise ValueError('Paged optimizer checkpoint parameter groups differ.')
        for old, new in zip(groups, self.optimizer.param_groups):
            if not isinstance(old, dict) or set(old) != set(new) - {'params'}:
                raise ValueError('Paged optimizer checkpoint hyperparameter groups differ.')
            import math
            if not isinstance(old.get('lr'), (int, float)) or isinstance(old['lr'], bool) or not math.isfinite(old['lr']) or not 0 <= old['lr'] <= 1:
                raise ValueError('Paged optimizer checkpoint learning rate is invalid.')
            for key in set(old) - {'lr'}:
                value = tuple(old[key]) if key == 'betas' and isinstance(old[key], list) else old[key]
                if value != new[key]:
                    raise ValueError('Paged optimizer checkpoint hyperparameter values differ.')
        expected_keys = set()
        for record, (group, index, name, parameter) in zip(records, parameters):
            if (record.get('group'), record.get('index'), record.get('name'), record.get('shape'), record.get('dtype')) != (group, index, name, list(parameter.shape), str(parameter.dtype)):
                raise ValueError('Paged optimizer checkpoint parameter layout differs.')
            keys, scalar = record.get('tensor_keys'), record.get('scalar')
            if not isinstance(keys, list) or len(set(keys)) != len(keys) or any(not isinstance(k, str) for k in keys) or not isinstance(scalar, dict):
                raise ValueError('Paged optimizer checkpoint state metadata is invalid.')
            if scalar and (set(scalar) != {'step'} or type(scalar['step']) is not int or scalar['step'] < 1):
                raise ValueError('Paged optimizer checkpoint step is invalid.')
            if bool(keys) != bool(scalar):
                raise ValueError('Paged optimizer checkpoint contains incomplete state.')
            expected_keys.update(f'{group}.{index}.{key}' for key in keys)
        with safe_open(str(source / 'state.safetensors'), framework='pt', device='cpu') as saved:
            if set(saved.keys()) != expected_keys:
                raise ValueError('Paged optimizer checkpoint tensor names differ.')
            self.optimizer.check_overrides()
            for record, (group, index, _, parameter) in zip(records, parameters):
                if not record['tensor_keys']:
                    continue
                self.optimizer.init_state(self.optimizer.param_groups[group], parameter, group, index)
                target = self.optimizer.state[parameter]
                if set(target) != set(record['tensor_keys']) | set(record['scalar']):
                    raise ValueError('Paged optimizer checkpoint tensor layout differs from this runtime.')
                for key in record['tensor_keys']:
                    value = saved.get_tensor(f'{group}.{index}.{key}')
                    if target[key].shape != value.shape or target[key].dtype != value.dtype:
                        raise ValueError('Paged optimizer checkpoint tensor dtype or shape differs.')
                    target[key].copy_(value)
                target.update(record['scalar'])
        for old, new in zip(groups, self.optimizer.param_groups):
            for key, value in old.items():
                new[key] = tuple(value) if key == 'betas' else value
        self.optimizer.initialized = True
        check_paged_finite(self.optimizer, [p for _, _, _, p in parameters])
