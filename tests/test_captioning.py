import base64
import json
from pathlib import Path
import sys
import threading
import time

import pytest
from PIL import Image

from sdxl_trainer.captioning import CaptionCancelled, CaptionConfig, CaptionError, CaptionServer, clean_caption
from sdxl_trainer.spelling import SpellChecker, SpellCheckUnavailable


FAKE_SERVER = '''#!PYTHON
import argparse, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p=argparse.ArgumentParser()
p.add_argument('--port',type=int)
p.add_argument('--host')
a,other=p.parse_known_args()
assert a.host == '127.0.0.1'
assert '--offline' in other and '--no-webui' in other
assert 'LLAMA_ARG_HF_REPO' not in os.environ
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send(self,body,status=200):
        body=json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == '/health': return self.send({'status':'ok'})
        if self.headers.get('Authorization') != 'Bearer '+os.environ['LLAMA_API_KEY']:
            return self.send({'error':'auth'},401)
        self.send({'data':[{'id':'sdxl-captioner'}]})
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert self.headers.get('Authorization') == 'Bearer '+os.environ['LLAMA_API_KEY']
        with open(os.environ['CAPTURE_FILE'],'w') as f: json.dump(body,f)
        time.sleep(float(os.environ.get('FAKE_DELAY','0')))
        self.send({'choices':[{'finish_reason':os.environ.get('FAKE_REASON','stop'),
                   'message':{'content':os.environ.get('FAKE_CONTENT','A red square.')}}]})
ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
'''


@pytest.fixture
def config(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server"
    exe.write_text(FAKE_SERVER.replace("PYTHON", sys.executable), encoding="utf-8")
    exe.chmod(0o755)
    model = tmp_path / "qwen; harmless.gguf"
    model.write_bytes(b"GGUFfake")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"GGUFfake")
    monkeypatch.setenv("CAPTURE_FILE", str(tmp_path / "captured.json"))
    monkeypatch.setenv("LLAMA_ARG_HF_REPO", "must/not-be-inherited")
    return CaptionConfig(server_executable=str(exe), model_path=str(model), mmproj_path=str(mmproj), startup_timeout=5)


def test_private_vision_request_uses_exact_png_and_all_controls(config, tmp_path):
    path = tmp_path / "Img-1.png"
    Image.new("RGB", (64, 64), "red").save(path)
    config.temperature = 0.37
    config.top_k = 19
    config.system_prompt = "Only describe pixels."
    with CaptionServer(config) as server:
        assert server.running
        assert server.caption(path) == "A red square."
        process = server._process
        payload = json.loads((tmp_path / "captured.json").read_text())
        uri = payload["messages"][1]["content"][0]["image_url"]["url"]
        assert base64.b64decode(uri.split(",", 1)[1]) == path.read_bytes()
        assert payload["messages"][0]["content"] == "Only describe pixels."
        assert payload["temperature"] == 0.37 and payload["top_k"] == 19
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["stream"] is False
    assert process.poll() is not None
    assert not server.running


def test_partial_caption_is_preserved_with_warning_at_continuation_limit(config, tmp_path, monkeypatch):
    path = tmp_path / "Img-1.png"
    Image.new("RGB", (64, 64)).save(path)
    monkeypatch.setenv("FAKE_REASON", "length")
    config.max_continuations = 0
    with CaptionServer(config) as server:
        result = server.caption(path)
    assert result == "A red square."
    assert result.truncated and "limit" in result.reason


def test_length_limited_caption_automatically_continues(config, tmp_path):
    path = tmp_path / "Img-1.png"
    Image.new("RGB", (64, 64)).save(path)
    requests = []
    responses = [
        {"choices": [{"finish_reason": "length", "message": {"content": "A red square"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": " on a blue background."}}]},
    ]
    with CaptionServer(config) as server:
        def request(method, url, payload, timeout):
            import copy
            requests.append(copy.deepcopy(payload))
            return responses.pop(0)
        server._request = request
        result = server.caption(path)
    assert result == "A red square on a blue background."
    assert not result.truncated
    assert len(requests) == 2
    assert len(requests[1]["messages"]) == 3
    assert requests[1]["messages"][-1] == {"role": "assistant", "content": "A red square"}


def test_failed_continuation_preserves_usable_partial_caption(config, tmp_path):
    path = tmp_path / "Img-1.png"
    Image.new("RGB", (64, 64)).save(path)
    responses = iter([{"choices": [{"finish_reason": "length", "message": {"content": "A red square"}}]}, CaptionError("context full")])
    with CaptionServer(config) as server:
        def request(*args):
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value
        server._request = request
        result = server.caption(path)
    assert result == "A red square"
    assert result.truncated and "context full" in result.reason


def test_cancel_generation_unloads_backend(config, tmp_path, monkeypatch):
    path = tmp_path / "Img-1.png"
    Image.new("RGB", (64, 64)).save(path)
    monkeypatch.setenv("FAKE_DELAY", "30")
    server = CaptionServer(config).start()
    event = threading.Event()
    timer = threading.Timer(0.3, event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(CaptionCancelled):
            server.caption(path, event)
        assert time.monotonic() - started < 4
        assert not server.running
    finally:
        timer.cancel()
        server.stop()


def test_remote_models_and_invalid_controls_are_rejected(config):
    config.model_path = "https://example.com/model.gguf"
    config.temperature = float("nan")
    config.ubatch_size = config.batch_size + 1
    config.flash_attention = "invalid"
    errors = "\n".join(config.validate())
    assert "local GGUF" in errors and "finite number" in errors
    assert "micro-batch" in errors and "Flash attention" in errors
    with pytest.raises(CaptionError):
        CaptionServer(config).start()


def test_command_never_interprets_shell_metacharacters(config):
    command = CaptionServer(config).command(12345)
    assert "qwen; harmless.gguf" in command[command.index("--model") + 1]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--offline" in command
    assert "--no-webui" in command


@pytest.mark.parametrize("text,result", [
    ("<think>analysis</think> A tree.", "A tree."),
    ("analysis from template</think>A tree.", "A tree."),
    ("A tree.", "A tree."),
])
def test_clean_reasoning(text, result):
    assert clean_caption(text) == result


@pytest.mark.parametrize("text", ["", "<think>unfinished", "<think>only analysis</think>", "\x00"])
def test_empty_or_broken_reasoning_is_rejected(text):
    with pytest.raises(CaptionError):
        clean_caption(text)


def test_missing_spell_dictionary_is_explicit():
    checker = SpellChecker("zz_NONEXISTENT")
    assert not checker.available
    assert checker.reason
    with pytest.raises(SpellCheckUnavailable):
        checker.check("hello")
    with pytest.raises(SpellCheckUnavailable):
        checker.misspellings("hello")
    checker.close()


def test_real_offline_spellcheck_when_available():
    checker = SpellChecker()
    if not checker.available:
        pytest.skip(checker.reason)
    try:
        assert checker.check("photograph")
        assert not checker.check("photogrraph")
        assert "photograph" in checker.suggest("photogrraph")
        assert checker.misspellings("A photogrraph.") == [(2, 11, "photogrraph")]
        checker.ignore("photogrraph")
        assert checker.check("photogrraph")
    finally:
        checker.close()


@pytest.mark.parametrize(('parts', 'expected'), [
    (['water', 'fall.'], 'waterfall.'),
    (['Hello', ', world.'], 'Hello, world.'),
    (['A square  ', 'and a circle.'], 'A square  and a circle.'),
    (['A wave', ' of light.'], 'A wave of light.'),
    (['ha', 'ha'], 'haha'),
    (['water', '<think>\n\n</think>\n\nfall'], 'waterfall'),
])
def test_continuations_preserve_exact_segment_boundaries(config, tmp_path, parts, expected):
    path = tmp_path / 'Img-1.png'
    Image.new('RGB', (64, 64)).save(path)
    responses = iter([
        {'choices':[{'finish_reason':'length', 'message':{'content':parts[0]}}]},
        {'choices':[{'finish_reason':'stop', 'message':{'content':parts[1]}}]},
    ])
    with CaptionServer(config) as server:
        server._request = lambda *args: next(responses)
        result = server.caption(path)
    assert result == expected
    assert not result.truncated


def test_multiple_prefill_continuations_use_the_complete_unmodified_prefix(config, tmp_path):
    import copy
    path = tmp_path / 'Img-1.png'
    Image.new('RGB', (64,64)).save(path)
    calls = []
    replies = iter([
        {'choices':[{'finish_reason':'length','message':{'content':'A water'}}]},
        {'choices':[{'finish_reason':'length','message':{'content':'fall,'}}]},
        {'choices':[{'finish_reason':'stop','message':{'content':' at dusk.'}}]},
    ])
    with CaptionServer(config) as server:
        def request(_method, _url, payload, _timeout):
            calls.append(copy.deepcopy(payload))
            return next(replies)
        server._request = request
        result = server.caption(path)
    assert result == 'A waterfall, at dusk.'
    assert not result.truncated
    assert [request['messages'][-1].get('content') for request in calls[1:]] == ['A water', 'A waterfall,']
    assert all(len(request['messages']) == 3 for request in calls[1:])


def test_prefill_echo_is_not_duplicated_and_boundaries_are_preserved(config, tmp_path):
    import copy
    path = tmp_path / 'Img-1.png'
    Image.new('RGB', (64, 64)).save(path)
    requests = []
    replies = iter([
        {'choices': [{'finish_reason':'length', 'message':{'content':'A water'}}]},
        {'choices': [{'finish_reason':'length', 'message':{'content':'A waterfall,'}}]},
        {'choices': [{'finish_reason':'stop', 'message':{'content':'A waterfall, at dusk.'}}]},
    ])
    with CaptionServer(config) as server:
        def request(_method, _url, payload, _timeout):
            requests.append(copy.deepcopy(payload))
            return next(replies)
        server._request = request
        result = server.caption(path)
    assert result == 'A waterfall, at dusk.'
    assert not result.truncated
    assert requests[1]['messages'][-1]['content'] == 'A water'
    assert requests[2]['messages'][-1]['content'] == 'A waterfall,'
