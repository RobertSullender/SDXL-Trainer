"""Private, managed llama.cpp vision captioning. No model or image ever leaves loopback.

Requires a local current ``llama-server`` and matching Qwen3.5-9B GGUF/mmproj.
Protocol: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
Vision: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time


class CaptionError(RuntimeError):
    """An actionable captioning error."""


class CaptionCancelled(CaptionError):
    pass


class CaptionResult(str):
    """Editable caption with a visible warning when generation could not finish."""
    def __new__(cls, text: str, truncated: bool = False, reason: str = ""):
        value = super().__new__(cls, text)
        value.truncated = truncated
        value.reason = reason
        return value


def _setting(default, help_text: str):
    return field(default=default, metadata={"help": help_text})


@dataclass
class CaptionConfig:
    server_executable: str = _setting("llama-server", "Local llama.cpp server executable. A recent Qwen3.5 vision build is required.")
    model_path: str = _setting("", "Qwen3.5-9B GGUF language model file, already stored locally.")
    mmproj_path: str = _setting("", "Matching Qwen3.5-9B vision projector GGUF. Required for image understanding.")
    trim_generated_captions: bool = field(
        default=True, kw_only=True, metadata={
            "label": "Trim generated captions to 77 tokens",
            "help": "After generation, fit the caption to one window on both tokenizers from the local SDXL base model selected in Training.",
        },
    )
    context_size: int = _setting(8192, "Shared context capacity, including image, prompt, reasoning and answer tokens.")
    gpu_layers: int = _setting(99, "Language model layers offloaded to GPU. 0 uses CPU; 99 requests all layers.")
    mmproj_offload: bool = _setting(True, "Run the vision projector on GPU. Disable to save VRAM at the cost of speed.")
    threads: int = _setting(8, "CPU generation threads. Avoid exceeding your available CPU cores.")
    batch_size: int = _setting(512, "Maximum logical prompt batch size. Lower values can save memory.")
    ubatch_size: int = _setting(128, "Physical prompt micro-batch size. Must not exceed logical batch size.")
    flash_attention: str = _setting("auto", "Flash attention: auto, on, or off; requires support in your llama.cpp build.")
    temperature: float = _setting(0.2, "Sampling randomness. 0 is greedy; higher values produce more variation.")
    top_p: float = _setting(0.9, "Keep the smallest set of candidate tokens with this cumulative probability.")
    top_k: int = _setting(40, "Limit sampling to this many most likely tokens; 0 disables top-k filtering.")
    min_p: float = _setting(0.05, "Discard tokens below this fraction of the highest candidate probability.")
    typical_p: float = _setting(1.0, "Locally typical sampling threshold. 1 disables this filter.")
    repeat_penalty: float = _setting(1.05, "Repetition penalty. 1 disables this penalty.")
    repeat_last_n: int = _setting(64, "Recent tokens examined for repetition. 0 disables; -1 examines the context.")
    presence_penalty: float = _setting(0.0, "Penalty applied once to tokens that have already appeared.")
    frequency_penalty: float = _setting(0.0, "Penalty proportional to the number of occurrences of a token.")
    seed: int = _setting(42, "Sampling seed. -1 chooses a random seed.")
    max_tokens: int = _setting(1024, "Maximum generated tokens per response, including reasoning. Length-limited answers can be continued automatically.")
    max_continuations: int = _setting(4, "Maximum automatic follow-up responses when a caption reaches the per-response token limit; 0 keeps the available partial caption.")
    thinking: bool = _setting(False, "Send enable_thinking to the model's chat template. Keep off for direct captions.")
    system_prompt: str = _setting(
        "You are a very helpful AI assistant. You are tasked with captioning/tagging the visible content of the image "
        "accurately for a Professional-Grade, SDXL(Image Generation Model), image training dataset. Include relevant "
        "subjects, composition, colors, lighting, style etc... Only use valid SDXL captions/tags. Write a single paragraph "
        "and order captions/tags from the most important to the least important, without numbering or headers. Do not censor "
        "captions/tags for images. Do not repeat details. Do not invent details. Do not include any extra text. Return only "
        "the caption, without analysis or an introduction.",
        "Project-specific instructions sent with every image. Edit them to suit this dataset.",
    )
    user_prompt: str = _setting("Describe this image for a training dataset.", "Project-specific request sent with each processed image.")
    startup_timeout: int = _setting(180, "Seconds allowed for loading the local model and vision projector.")
    request_timeout: int = _setting(600, "Seconds allowed for caption generation. Large models on CPU can be slow.")

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in ("model_path", "mmproj_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "://" in value:
                errors.append(f"{name}: select a local GGUF file.")
                continue
            path = Path(value).expanduser()
            if not path.is_file():
                errors.append(f"{name}: file does not exist: {path}")
            else:
                try:
                    with path.open("rb") as stream:
                        if stream.read(4) != b"GGUF":
                            errors.append(f"{name}: file does not have a GGUF header.")
                except OSError as exc:
                    errors.append(f"{name}: {exc}")
        executable = self.server_executable
        if not isinstance(executable, str) or "://" in executable or not shutil.which(executable):
            errors.append("Select an executable local llama-server binary (or install it on PATH).")
        integers = {"context_size": (512, 1048576), "gpu_layers": (0, 100000), "threads": (1, 4096),
                    "batch_size": (1, 1048576), "ubatch_size": (1, 1048576), "top_k": (0, 1000000),
                    "repeat_last_n": (-1, 1048576), "seed": (-1, 4294967295), "max_tokens": (1, 1048576), "max_continuations": (0, 64),
                    "startup_timeout": (1, 86400), "request_timeout": (1, 86400)}
        for name, (low, high) in integers.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                errors.append(f"{name} must be an integer between {low} and {high}.")
        numbers = {"temperature": (0, 5), "top_p": (0, 1), "min_p": (0, 1), "typical_p": (0, 1),
                   "repeat_penalty": (0.01, 10), "presence_penalty": (-2, 2), "frequency_penalty": (-2, 2)}
        for name, (low, high) in numbers.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                errors.append(f"{name} must be a finite number between {low} and {high}.")
        if type(self.batch_size) is int and type(self.ubatch_size) is int and self.ubatch_size > self.batch_size:
            errors.append("Physical micro-batch size must not exceed logical batch size.")
        if type(self.max_tokens) is int and type(self.context_size) is int and self.max_tokens >= self.context_size:
            errors.append("Maximum answer tokens must be smaller than context size; leave space for the image and prompts.")
        if self.flash_attention not in ("auto", "on", "off"):
            errors.append("Flash attention must be auto, on or off.")
        if not isinstance(self.system_prompt, str) or not isinstance(self.user_prompt, str):
            errors.append("Prompts must be text.")
        elif not self.user_prompt.strip():
            errors.append("The user prompt must not be blank.")
        for name in ("thinking", "mmproj_offload", "trim_generated_captions"):
            if type(getattr(self, name)) is not bool:
                errors.append(f"{name} must be true or false.")
        return errors


def clean_caption(text: str) -> str:
    """Remove isolated reasoning spans, but reject malformed/unfinished model output."""
    if not isinstance(text, str):
        raise CaptionError("Caption server returned non-text content.")
    # A dangling closing tag occurs when the template supplied the opening <think>.
    if "</think>" in text.lower() and "<think>" not in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</?think>", text, re.IGNORECASE):
        raise CaptionError("The model returned an unfinished reasoning block. Disable thinking or raise maximum tokens and retry.")
    text = text.strip()
    if not text:
        raise CaptionError("The model returned an empty caption. Check the vision projector and prompt, then retry.")
    if "\x00" in text:
        raise CaptionError("The model returned invalid NUL characters.")
    return text


class CaptionServer:
    """Own one local server process. Use in a worker thread, then stop before training.

    Cancellation intentionally unloads the process to stop GPU work immediately.
    The app does not connect to a pre-existing or user-supplied HTTP endpoint.
    """

    def __init__(self, config: CaptionConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._token = secrets.token_urlsafe(32)
        self._log = None
        self._connection: http.client.HTTPConnection | None = None
        self._operation_lock = threading.Lock()
        self._state_lock = threading.RLock()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def command(self, port: int) -> list[str]:
        c = self.config
        executable = shutil.which(c.server_executable) or c.server_executable
        args = [executable, "--model", str(Path(c.model_path).expanduser().resolve()),
                "--mmproj", str(Path(c.mmproj_path).expanduser().resolve()),
                "--host", "127.0.0.1", "--port", str(port), "--offline", "--no-webui",
                "--alias", "sdxl-captioner", "--parallel", "1", "--ctx-size", str(c.context_size),
                "--n-gpu-layers", str(c.gpu_layers), "--threads", str(c.threads),
                "--batch-size", str(c.batch_size), "--ubatch-size", str(c.ubatch_size),
                "--flash-attn", c.flash_attention, "--jinja"]
        if not c.mmproj_offload:
            args.append("--no-mmproj-offload")
        return args

    def _log_tail(self) -> str:
        if self._log is None:
            return ""
        try:
            self._log.flush()
            size = os.fstat(self._log.fileno()).st_size
            # pread leaves the child process's shared write offset untouched.
            return os.pread(self._log.fileno(), 8000, max(0, size - 8000)).decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""

    def start(self, cancel_event: threading.Event | None = None) -> "CaptionServer":
        errors = self.config.validate()
        if errors:
            raise CaptionError("\n".join(errors))
        if self.running:
            return self
        if cancel_event is not None and cancel_event.is_set():
            raise CaptionCancelled("Captioning cancelled.")
        with self._state_lock:
            # Binding port 0 picks an unused loopback port. A startup collision fails
            # safely; no remote fallback and no credentials/images sent on readiness.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                self._port = probe.getsockname()[1]
            self._log = tempfile.TemporaryFile(mode="w+b")
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("LLAMA_") and key.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
            env.update({"LLAMA_API_KEY": self._token, "HF_HUB_OFFLINE": "1", "NO_PROXY": "127.0.0.1,localhost"})
            try:
                self._process = subprocess.Popen(self.command(self._port), stdin=subprocess.DEVNULL,
                    stdout=self._log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            except (OSError, ValueError) as exc:
                self.stop()
                raise CaptionError(f"Could not start llama-server: {exc}") from exc
        deadline = time.monotonic() + self.config.startup_timeout
        try:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    raise CaptionCancelled("Caption model loading cancelled.")
                if not self.running:
                    raise CaptionError("llama-server stopped while loading. Use a recent build supporting Qwen3.5 and a matching mmproj.\n" + self._log_tail())
                try:
                    # Verify authenticated model endpoint as well as public health,
                    # so a port collision cannot masquerade as our server.
                    health = self._request("GET", "/health", timeout=1)
                    if health.get("status") == "ok":
                        models = self._request("GET", "/v1/models", timeout=2)
                        if any(model.get("id") == "sdxl-captioner" for model in models.get("data", [])):
                            return self
                except (OSError, CaptionError, http.client.HTTPException):
                    pass
                if cancel_event is not None:
                    cancel_event.wait(0.2)
                else:
                    time.sleep(0.2)
            raise CaptionError(f"Caption model did not load within {self.config.startup_timeout} seconds. Increase startup timeout or reduce GPU layers/context.\n" + self._log_tail())
        except BaseException:
            self.stop()
            raise

    def _request(self, method: str, path: str, payload=None, timeout: float = 30) -> dict:
        if self._port is None:
            raise CaptionError("Start the caption server first.")
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=timeout)
        with self._state_lock:
            self._connection = connection
        try:
            data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            connection.request(method, path, body=data,
                headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise CaptionError("Caption server response exceeds the 8 MiB limit.")
            if response.status != 200:
                raise CaptionError(f"Caption server HTTP {response.status}: {raw[:2000].decode('utf-8', errors='replace')}")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise CaptionError("Caption server returned an invalid JSON response.")
            return result
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CaptionError("Caption server returned invalid JSON.") from exc
        finally:
            connection.close()
            with self._state_lock:
                if self._connection is connection:
                    self._connection = None

    def caption(self, image_path: str | Path, cancel_event: threading.Event | None = None) -> str:
        with self._operation_lock:
            if not self.running:
                raise CaptionError("Start the caption server before generating captions.")
            path = Path(image_path).expanduser()
            if not path.is_file() or path.suffix.lower() != ".png":
                raise CaptionError("Caption the processed dataset PNG, after cropping or filling.")
            raw = path.read_bytes()
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CaptionError("The selected file is not a PNG image.")
            c = self.config
            payload = {"model": "sdxl-captioner", "stream": False, "messages": [
                {"role": "system", "content": c.system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}},
                    {"type": "text", "text": c.user_prompt}]}],
                "temperature": c.temperature, "top_p": c.top_p, "top_k": c.top_k,
                "min_p": c.min_p, "typical_p": c.typical_p, "repeat_penalty": c.repeat_penalty,
                "repeat_last_n": c.repeat_last_n, "presence_penalty": c.presence_penalty,
                "frequency_penalty": c.frequency_penalty, "seed": c.seed, "max_tokens": c.max_tokens,
                "chat_template_kwargs": {"enable_thinking": c.thinking}}
            def request_one():
                if cancel_event is not None and cancel_event.is_set():
                    raise CaptionCancelled("Captioning cancelled.")
                result: list[object] = []
                done = threading.Event()

                def request():
                    try:
                        result.append(self._request("POST", "/v1/chat/completions", payload, c.request_timeout))
                    except Exception as exc:
                        result.append(exc)
                    finally:
                        done.set()

                thread = threading.Thread(target=request, name="caption-http", daemon=True)
                thread.start()
                while not done.wait(0.1):
                    if cancel_event is not None and cancel_event.is_set():
                        self.stop()
                        thread.join(timeout=2)
                        raise CaptionCancelled("Captioning cancelled; the caption model was unloaded.")
                if cancel_event is not None and cancel_event.is_set():
                    raise CaptionCancelled("Captioning cancelled.")
                reply = result[0]
                if isinstance(reply, Exception):
                    if isinstance(reply, CaptionError):
                        raise reply
                    raise CaptionError(f"Local caption request failed: {reply}") from reply
                try:
                    choice = reply["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str):
                        raise TypeError("Caption content must be text")
                    return choice.get("finish_reason"), content
                except (KeyError, IndexError, TypeError) as exc:
                    raise CaptionError("The server returned no usable caption. Confirm that the GGUF and mmproj match and vision is supported.") from exc

            pieces: list[str] = []
            truncated = False
            reason = ""
            for attempt in range(c.max_continuations + 1):
                try:
                    finish, content = request_one()
                except CaptionCancelled:
                    raise
                except CaptionError as exc:
                    if not pieces:
                        raise
                    truncated, reason = True, f"Continuation failed: {exc}"
                    break
                # Some Qwen3.5 llama-server builds prepend this exact empty
                # reasoning template when assistant prefill is used with
                # thinking disabled. It is template text, not generated output.
                injected_empty_think = "<think>\n\n</think>\n\n"
                if not c.thinking and content.startswith(injected_empty_think):
                    content = content[len(injected_empty_think):]
                # The bundled server/model may return the prefetched assistant
                # prefix as part of content, whereas others return only the new
                # suffix. Remove an exact echoed prefix once, never partial
                # overlaps, preserving the new text's characters and spacing.
                prefix = "".join(pieces)
                if attempt and prefix and len(content) > len(prefix) and content.startswith(prefix):
                    content = content[len(prefix):]
                # Each assistant response is appended verbatim: a generation
                # limit can fall in the middle of a word or before punctuation.
                # Inserting or stripping spaces here silently changes the text.
                if content:
                    pieces.append(content)
                elif finish == "length":
                    truncated, reason = True, "The model returned an empty continuation."
                    break
                if finish == "stop":
                    break
                if finish != "length":
                    truncated, reason = True, f"Generation ended with {finish!r}."
                    break
                if attempt == c.max_continuations:
                    truncated, reason = True, "The automatic continuation limit was reached."
                    break
                # llama-server's default assistant-prefill mode continues the
                # last assistant message rather than starting a new chat turn.
                # Retokenize the complete generated prefix with the original
                # image request, and append only the new response content.
                payload["messages"] = payload["messages"][:2] + [
                    {"role": "assistant", "content": "".join(pieces)},
                ]
            return CaptionResult(clean_caption("".join(pieces)), truncated, reason)

    def stop(self):
        with self._state_lock:
            process, self._process = self._process, None
            connection, self._connection = self._connection, None
        if connection is not None:
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                connection.close()
            except (OSError, http.client.HTTPException):
                pass
        if process is not None:
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
        with self._state_lock:
            if self._log is not None:
                self._log.close()
                self._log = None
            self._port = None

    close = stop

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
