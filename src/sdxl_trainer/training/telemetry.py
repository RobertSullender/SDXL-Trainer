"""Low-overhead training phase timing and local NVIDIA GPU sampling.

Wall times describe how long the Python thread spent in each scope. CUDA work is
asynchronous, so these are not exclusive GPU timings. Optional CUDA events measure
elapsed time on the current GPU stream, synchronizing once per completed step.
GPU utilization is sampled independently and is not a substitute for a profiler.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import csv
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Callable


def _number(value: str) -> float | None:
    try:
        number = float(value.strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _system_memory(read_text=None) -> dict:
    """Read host-wide available RAM and swap plus this worker's swap, in MiB.

    Linux /proc reports these values in kB (1024 bytes). Missing fields are
    represented as None; monitoring must work even when /proc is unavailable.
    """
    if read_text is None:
        read_text = lambda path: Path(path).read_text()

    def fields(path):
        try:
            result = {}
            for line in read_text(path).splitlines():
                key, separator, value = line.partition(":")
                parts = value.split()
                if separator and len(parts) == 2 and parts[1] == "kB":
                    number = _number(parts[0])
                    if number is not None:
                        result[key] = number / 1024
            return result
        except (OSError, ValueError, TypeError):
            return {}

    host = fields("/proc/meminfo")
    process = fields("/proc/self/status")
    swap_total = host.get("SwapTotal")
    swap_free = host.get("SwapFree")
    return {"mem_available_mib": host.get("MemAvailable"),
            "swap_used_mib": max(0.0, swap_total - swap_free) if swap_total is not None and swap_free is not None else None,
            "swap_total_mib": swap_total,
            "process_swap_mib": process.get("VmSwap")}


def _device_uuid(torch_module, device: str) -> str | None:
    """Resolve a CUDA-visible logical device to an unambiguous physical UUID."""
    try:
        props = torch_module.cuda.get_device_properties(torch_module.device(device))
        value = str(getattr(props, "uuid", "") or "").strip()
    except Exception:
        return None
    if not value:
        return None
    return value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value


class GpuSampler:
    """Sample exactly the CUDA device's UUID with the local nvidia-smi program.

    The callback runs in a daemon thread and must not manipulate Qt widgets.
    Query failures never interrupt training. Samples are bounded in memory.
    """

    def __init__(self, uuid: str | None, *, interval: float = 2.0,
                 on_sample: Callable[[dict], None] | None = None,
                 clock=time.monotonic, runner=subprocess.run, executable=None,
                 memory_reader=_system_memory):
        self.uuid = uuid
        self.interval = max(0.25, float(interval))
        self.on_sample = on_sample
        self.clock = clock
        self.runner = runner
        self.memory_reader = memory_reader
        self.executable = executable if executable is not None else shutil.which("nvidia-smi")
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.samples = deque(maxlen=4096)
        self.thread = None

    def _callback(self, sample):
        if self.on_sample:
            try:
                self.on_sample(dict(sample))
            except Exception:
                # A monitoring callback cannot break the training worker.
                pass

    def _unavailable(self, message):
        self._callback({"available": False, "reason": message, "device_name": None,
                        "utilization_percent": None, "memory_used_mib": None,
                        "memory_total_mib": None, "sample_time": self.clock(),
                        "system_memory": self._memory()})

    def _memory(self):
        try:
            return self.memory_reader()
        except Exception:
            return {}

    def sample_once(self) -> dict | None:
        if not self.uuid:
            self._unavailable("Cannot map the selected CUDA device to a GPU UUID.")
            return None
        if not self.executable:
            self._unavailable("nvidia-smi is unavailable.")
            return None
        command = [self.executable, "-i", self.uuid,
                   "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                   "--format=csv,noheader,nounits"]
        try:
            result = self.runner(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=3, check=True)
            rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
            if len(rows) != 1 or len(rows[0]) != 7:
                raise ValueError("Unexpected nvidia-smi response.")
            gpu_uuid, name, utilization, used, total, temperature, power = [x.strip() for x in rows[0]]
            if gpu_uuid != self.uuid:
                raise ValueError("NVIDIA reported a different GPU UUID.")
            sample = {"available": True, "device_name": name, "gpu_uuid": gpu_uuid,
                      "utilization_percent": _number(utilization), "memory_used_mib": _number(used),
                      "memory_total_mib": _number(total), "temperature_c": _number(temperature),
                      "power_w": _number(power), "sample_time": self.clock(),
                      "system_memory": self._memory()}
            if sample["utilization_percent"] is None and sample["memory_used_mib"] is None:
                raise ValueError("GPU utilization and memory readings are unavailable.")
        except (OSError, ValueError, subprocess.SubprocessError, csv.Error) as exc:
            self._unavailable(f"GPU monitoring unavailable: {exc}")
            return None
        with self.lock:
            self.samples.append(sample)
        self._callback(sample)
        return sample

    def start(self):
        if self.thread is not None:
            return
        self.stop_event.clear()
        def poll():
            while not self.stop_event.is_set():
                self.sample_once()
                if self.stop_event.wait(self.interval):
                    break
        self.thread = threading.Thread(target=poll, name="sdxl-gpu-monitor", daemon=True)
        self.thread.start()

    def summary(self, start: float, end: float) -> dict:
        with self.lock:
            samples = [s for s in self.samples if start <= s["sample_time"] <= end]
        if not samples:
            return {"sample_count": 0}
        utils = [s["utilization_percent"] for s in samples if s["utilization_percent"] is not None]
        memory = [s["memory_used_mib"] for s in samples if s["memory_used_mib"] is not None]
        return {"sample_count": len(samples), "device_name": samples[-1]["device_name"],
                "utilization_mean_percent": sum(utils) / len(utils) if utils else None,
                "utilization_peak_percent": max(utils) if utils else None,
                "memory_used_peak_mib": max(memory) if memory else None,
                "memory_total_mib": samples[-1]["memory_total_mib"]}

    def latest(self) -> dict | None:
        with self.lock:
            return dict(self.samples[-1]) if self.samples else None

    def close(self):
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=4)
        self.thread = None


class StepTelemetry:
    """Collect one optimizer step's CPU wall timings and optional GPU stream timings."""

    def __init__(self, device: str, *, gpu_enabled: bool = True, sample_interval: float = 2.0,
                 synchronize_gpu: bool = True, monitor_gpu: bool = True,
                 on_sample: Callable[[dict], None] | None = None,
                 clock=time.monotonic, torch_module=None, sampler_factory=GpuSampler):
        self.device = str(device)
        self.clock = clock
        self.synchronize_gpu = synchronize_gpu
        self.torch = torch_module
        self.use_events = False
        self.monitor = None
        self.walls = {}
        self.events = {}
        self.step_start = None
        if self.device.startswith("cuda"):
            try:
                if self.torch is None:
                    import torch
                    self.torch = torch
                if self.torch.cuda.is_available():
                    self.use_events = bool(gpu_enabled and synchronize_gpu)
                    if monitor_gpu:
                        self.monitor = sampler_factory(_device_uuid(self.torch, self.device),
                                                       interval=sample_interval, on_sample=on_sample, clock=clock)
            except Exception:
                # Telemetry is optional; training preflight handles unavailable CUDA.
                self.use_events = False
                self.monitor = None

    def start(self):
        if self.monitor:
            self.monitor.start()
        return self

    def begin_step(self):
        self.walls = {}
        self.events = {}
        self.step_start = self.clock()

    @contextmanager
    def phase(self, name: str, *, gpu: bool = False):
        if self.step_start is None:
            raise RuntimeError("Call begin_step() before timing a phase.")
        start_event = end_event = None
        if gpu and self.use_events:
            try:
                start_event = self.torch.cuda.Event(enable_timing=True)
                end_event = self.torch.cuda.Event(enable_timing=True)
                start_event.record()
            except Exception:
                # Disable event timings rather than affect model execution.
                self.use_events = False
                start_event = end_event = None
        started = self.clock()
        try:
            yield
        finally:
            elapsed = max(0.0, self.clock() - started)
            self.walls[name] = self.walls.get(name, 0.0) + elapsed
            if start_event is not None and end_event is not None:
                try:
                    end_event.record()
                    self.events.setdefault(name, []).append((start_event, end_event))
                except Exception:
                    self.use_events = False
                    self.events.clear()

    def finish_step(self, images: int, step: int) -> dict:
        if self.step_start is None:
            raise RuntimeError("Call begin_step() before finishing a step.")
        gpu_times = {}
        if self.use_events and self.events:
            try:
                # One synchronization at the step boundary; no per-phase stalls.
                self.torch.cuda.synchronize(self.device)
                for name, pairs in self.events.items():
                    gpu_times[name] = round(sum(start.elapsed_time(end) for start, end in pairs) / 1000, 6)
            except Exception:
                gpu_times = {}
                self.use_events = False
        ended = self.clock()
        duration = max(0.0, ended - self.step_start)
        result = {"step": int(step), "step_seconds": round(duration, 6),
                  "images_per_second": round(images / duration, 6) if duration > 0 else None,
                  "phase_wall_seconds": {k: round(v, 6) for k, v in self.walls.items()},
                  "phase_gpu_stream_seconds": gpu_times,
                  "gpu": self.monitor.summary(self.step_start, ended) if self.monitor else {"sample_count": 0}}
        self.step_start = None
        return result

    def latest(self) -> dict | None:
        return self.monitor.latest() if self.monitor else None

    def close(self):
        if self.monitor:
            self.monitor.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
