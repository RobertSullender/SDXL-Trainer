"""Use a separate local Python runtime for ML without making the desktop heavyweight."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from .paths import offline_environment


class RemoteTokenCounter:
    """One local process loads both tokenizers once, then handles JSONL requests."""

    def __init__(self, python: str, model_path: str, startup_timeout: float = 60,
                 request_timeout: float = 30):
        self._buffer = b""
        self.request_timeout = request_timeout
        self.process = subprocess.Popen(
            [python, "-m", "sdxl_trainer.runtime", "tokens", "--model", model_path],
            env=offline_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
        )
        try:
            result = self._read(startup_timeout)
            if result.get("error"):
                raise RuntimeError(result["error"])
            if result.get("ready") is not True or result.get("max_length") != 77:
                raise RuntimeError("Tokenizer runtime returned an invalid SDXL initialization response.")
            self.max_length = 77
            self.metadata = result.get("metadata", {})
            if not isinstance(self.metadata, dict):
                raise RuntimeError("Tokenizer runtime returned invalid tokenizer metadata.")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout=None):
        deadline = time.monotonic() + (self.request_timeout if timeout is None else timeout)
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Tokenizer runtime timed out. Check the Python executable and local tokenizer files.")
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Tokenizer runtime exited. Check the Python runtime and local model tokenizers.")
            self._buffer += chunk
            if len(self._buffer) > 1024 * 1024:
                raise RuntimeError("Tokenizer runtime response exceeded the protocol size limit.")
        line, self._buffer = self._buffer.split(b"\n", 1)
        result = json.loads(line)
        if not isinstance(result, dict):
            raise RuntimeError("Tokenizer runtime returned an invalid JSON response.")
        return result

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._read()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def count(self, text: str):
        from .tokens import TokenCounts
        return TokenCounts(**self._request({"text": text}))

    def truncate(self, text: str) -> str:
        result = self._request({"action": "truncate", "text": text})
        prefix = result.get("text")
        if not isinstance(prefix, str) or not text.startswith(prefix):
            raise RuntimeError("Tokenizer runtime returned an invalid shortened caption.")
        return prefix

    def close(self):
        process = getattr(self, "process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process:
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["tokens", "diagnose"])
    parser.add_argument("--model", default="")
    args = parser.parse_args(argv)
    if args.action == "diagnose":
        import importlib.util
        result = {"python": sys.version, "executable": sys.executable,
                  "modules": {m: importlib.util.find_spec(m) is not None for m in
                              ["torch", "diffusers", "transformers", "peft", "safetensors", "PIL"]}}
        try:
            import torch
            result.update(torch=torch.__version__, cuda=torch.version.cuda,
                          cuda_available=torch.cuda.is_available(),
                          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          cuda_architectures=torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
        except ImportError:
            pass
        print(json.dumps(result, indent=2))
        return 0
    try:
        from .tokens import TokenCounter
        counter = TokenCounter.from_model(Path(args.model))
        print(json.dumps({"ready": True, "max_length": 77, "metadata": counter.metadata}), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                text = request["text"]
                action = request.get("action", "count")
                if action == "count":
                    result = counter.count(text).to_dict()
                elif action == "truncate":
                    result = {"text": counter.truncate(text)}
                else:
                    raise ValueError(f"Unknown tokenizer action: {action!r}")
                print(json.dumps(result), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
