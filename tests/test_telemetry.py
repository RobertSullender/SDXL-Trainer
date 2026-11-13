"""Telemetry should expose useful measurements without depending on a GPU."""
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from sdxl_trainer.training.telemetry import GpuSampler, StepTelemetry, _device_uuid, _system_memory


def test_system_memory_samples_host_and_worker_without_confusing_units():
    files = {"/proc/meminfo": "MemAvailable: 1048576 kB\nSwapTotal: 2097152 kB\nSwapFree: 524288 kB\n",
             "/proc/self/status": "Name: trainer\nVmSwap: 2048 kB\n"}
    assert _system_memory(files.__getitem__) == {"mem_available_mib": 1024, "swap_used_mib": 1536,
                                                "swap_total_mib": 2048, "process_swap_mib": 2}


def test_system_memory_missing_or_malformed_values_are_nonfatal():
    def missing(path):
        if path.endswith("status"):
            raise OSError("not available")
        return "MemAvailable: N/A kB\nSwapTotal: 1024 kB\nSwapFree: 12 MB\n"
    assert _system_memory(missing) == {"mem_available_mib": None, "swap_used_mib": None,
                                       "swap_total_mib": 1, "process_swap_mib": None}


def test_cpu_step_timings_accumulate_and_do_not_import_cuda():
    times = iter([1.0, 2.0, 3.5, 4.0, 6.0, 7.0])
    timer = StepTelemetry("cpu", clock=lambda: next(times))
    timer.begin_step()
    with timer.phase("vae"):
        pass
    with timer.phase("vae"):
        pass
    result = timer.finish_step(images=4, step=9)
    assert result["step"] == 9
    assert result["step_seconds"] == 6
    assert result["phase_wall_seconds"] == {"vae": 3.5}
    assert result["phase_gpu_stream_seconds"] == {}
    assert result["images_per_second"] == pytest.approx(4 / 6, abs=1e-6)
    assert result["gpu"]["sample_count"] == 0


def test_phase_timing_requires_active_step_and_records_on_exception():
    clock = iter([1.0, 2.0, 3.0, 5.0])
    timer = StepTelemetry("cpu", clock=lambda: next(clock))
    with pytest.raises(RuntimeError, match="begin_step"):
        with timer.phase("vae"):
            pass
    timer.begin_step()
    with pytest.raises(ValueError):
        with timer.phase("vae"):
            raise ValueError("model error")
    result = timer.finish_step(1, 1)
    assert result["phase_wall_seconds"]["vae"] == 1.0
    with pytest.raises(RuntimeError, match="begin_step"):
        timer.finish_step(1, 2)


def test_cuda_events_sync_once_per_step_without_per_phase_stall():
    syncs = []
    events = []
    class Event:
        def __init__(self, enable_timing):
            assert enable_timing is True
            self.number = len(events)
            events.append(self)
        def record(self):
            pass
        def elapsed_time(self, other):
            return 250.0
    cuda = SimpleNamespace(is_available=lambda: True, Event=Event,
                           synchronize=lambda device: syncs.append(device),
                           get_device_properties=lambda device: SimpleNamespace(uuid="a-b-c"))
    torch = SimpleNamespace(cuda=cuda, device=lambda value: value)
    timer = StepTelemetry("cuda", torch_module=torch, monitor_gpu=False)
    timer.begin_step()
    with timer.phase("unet_forward", gpu=True):
        pass
    with timer.phase("unet_forward", gpu=True):
        pass
    with timer.phase("cpu_vae", gpu=False):
        pass
    assert syncs == []
    result = timer.finish_step(images=2, step=1)
    assert syncs == ["cuda"]
    assert result["phase_gpu_stream_seconds"] == {"unet_forward": 0.5}
    assert "cpu_vae" in result["phase_wall_seconds"]


def test_disabled_gpu_timing_never_synchronizes_or_records_events():
    cuda = SimpleNamespace(is_available=lambda: True,
        Event=lambda **kwargs: pytest.fail("events must be disabled"),
        synchronize=lambda device: pytest.fail("synchronization must be disabled"))
    timer = StepTelemetry("cuda", torch_module=SimpleNamespace(cuda=cuda), monitor_gpu=False,
                          synchronize_gpu=False)
    timer.begin_step()
    with timer.phase("unet", gpu=True):
        pass
    assert timer.finish_step(1, 1)["phase_gpu_stream_seconds"] == {}


def test_gpu_sampler_uses_exact_uuid_and_reports_sampled_metrics():
    calls, callbacks = [], []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="GPU-a-b-c, RTX 5080, 74, 1234, 16303, 58, 101.2\n")
    monitor = GpuSampler("GPU-a-b-c", executable="nvidia-smi", runner=run,
                         on_sample=callbacks.append, clock=lambda: 10.0,
                         memory_reader=lambda: {"swap_used_mib": 20.0})
    sample = monitor.sample_once()
    assert calls[0][1:3] == ["-i", "GPU-a-b-c"]
    assert sample["utilization_percent"] == 74
    assert sample["memory_used_mib"] == 1234
    assert sample["memory_total_mib"] == 16303
    assert sample["system_memory"]["swap_used_mib"] == 20
    assert callbacks == [sample]
    assert monitor.summary(9, 11)["utilization_mean_percent"] == 74
    assert monitor.summary(11, 12) == {"sample_count": 0}


def test_wrong_uuid_and_missing_monitor_are_unavailable_not_misreported():
    callbacks = []
    monitor = GpuSampler("GPU-real", executable="nvidia-smi", on_sample=callbacks.append,
        runner=lambda *a, **k: SimpleNamespace(stdout="GPU-other, other GPU, 50, 10, 100, 50, 10\n"))
    assert monitor.sample_once() is None
    assert monitor.sample_once() is None
    assert len(callbacks) == 2  # Host memory remains live even if the GPU monitor fails.
    assert callbacks[0]["available"] is False
    assert monitor.latest() is None
    assert GpuSampler(None, executable="nvidia-smi").sample_once() is None


def test_gpu_monitor_tolerates_missing_and_non_numeric_values():
    callback = []
    monitor = GpuSampler("GPU-x", executable="nvidia-smi", on_sample=callback.append,
        runner=lambda *a, **k: SimpleNamespace(stdout="GPU-x, GPU name, N/A, 10, 100, N/A, N/A\n"))
    result = monitor.sample_once()
    assert result["utilization_percent"] is None
    assert result["memory_used_mib"] == 10
    assert result["temperature_c"] is None
    monitor.runner = lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 3))
    assert monitor.sample_once() is None
    assert callback[-1]["available"] is False


def test_device_uuid_resolves_cuda_visible_logical_device_not_physical_ordinal():
    calls = []
    cuda = SimpleNamespace(get_device_properties=lambda index: calls.append(index) or SimpleNamespace(uuid="1234-abcd"))
    torch = SimpleNamespace(cuda=cuda, device=lambda value: "logical:" + value)
    assert _device_uuid(torch, "cuda:1") == "GPU-1234-abcd"
    assert calls == ["logical:cuda:1"]
    assert _device_uuid(SimpleNamespace(cuda=SimpleNamespace(), device=lambda x: x), "cuda") is None


def test_monitor_thread_stops_and_callbacks_do_not_raise():
    seen = threading.Event()
    def callback(sample):
        seen.set()
        raise ValueError("UI observer failed")
    monitor = GpuSampler("GPU-x", interval=0.25, executable="nvidia-smi", on_sample=callback,
        runner=lambda *a, **k: SimpleNamespace(stdout="GPU-x, GPU, 40, 10, 100, 45, 20\n"))
    monitor.start()
    assert seen.wait(2)
    thread = monitor.thread
    monitor.close()
    assert not thread.is_alive()
    assert monitor.latest()["utilization_percent"] == 40
