"""A tiny local server that speaks enough of the Gemini and Ollama HTTP APIs to test our clients."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.requests = []                    # (method, path, headers, body)
        self.gemini_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "text-embedding-x"]
        self.gemini_status = 200              # forced status for generateContent (0 = normal)
        self.gemini_reply = "Samosa and tea"
        self.gemini_delay = 0.0
        self.reject_thinking = set()          # e.g. {"minimal"} -> 400 mentioning thinking
        self.gemini_key = "test-key"
        self.ollama_models = [("llama3.2:3b", 2_000_000_000), ("tiny:1b", 700_000_000)]
        self.ollama_reply = "Tea at 4:30"
        self.ollama_delay = 0.0
        self.ollama_reject_think = False
        self.connections = 0


STATE = State()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"            # keep-alive, like the real servers

    def log_message(self, *a):
        pass

    def setup(self):
        super().setup()
        STATE.connections += 1

    def _send(self, status, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        STATE.requests.append(("GET", self.path, dict(self.headers), None))
        if self.path.startswith("/v1beta/models"):
            if self.headers.get("x-goog-api-key") != STATE.gemini_key:
                return self._send(403, {"error": {"message": "API key not valid"}})
            return self._send(200, {"models": [
                {"name": f"models/{m}", "supportedGenerationMethods": ["generateContent"] if "embedding" not in m else ["embedContent"]}
                for m in STATE.gemini_models]})
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": n, "size": s} for n, s in STATE.ollama_models]})
        self._send(404, {"error": "nope"})

    def do_POST(self):
        body = self._body()
        STATE.requests.append(("POST", self.path, dict(self.headers), body))
        if self.path.startswith("/v1beta/models/") and self.path.endswith(":generateContent"):
            model = self.path.split("/")[3].split(":")[0]
            if self.headers.get("x-goog-api-key") != STATE.gemini_key:
                return self._send(403, {"error": {"message": "API key not valid"}})
            if STATE.gemini_status not in (0, 200):
                hdr = {"Retry-After": "7"} if STATE.gemini_status == 429 else {}
                return self._send(STATE.gemini_status, {"error": {"message": f"forced {STATE.gemini_status}"}}, hdr)
            if model not in STATE.gemini_models:
                return self._send(404, {"error": {"message": f"models/{model} is not found"}})
            tc = body.get("generationConfig", {}).get("thinkingConfig") or {}
            lvl = tc.get("thinkingLevel")
            if lvl in STATE.reject_thinking:
                return self._send(400, {"error": {"message": f"Thinking level {lvl} is not supported for this model"}})
            time.sleep(STATE.gemini_delay)
            return self._send(200, {"candidates": [{"content": {"parts": [{"text": STATE.gemini_reply}]}, "finishReason": "STOP"}],
                                    "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 5, "thoughtsTokenCount": 0}})
        if self.path == "/api/chat":
            if not body.get("messages"):                                  # warm-up call
                return self._send(200, {"message": {"content": ""}, "done": True})
            if STATE.ollama_reject_think and "think" in body:
                return self._send(400, {"error": "\"model\" does not support thinking"})
            names = [n for n, _ in STATE.ollama_models]
            if body.get("model") not in names:
                return self._send(404, {"error": "model not found"})
            time.sleep(STATE.ollama_delay)
            return self._send(200, {"message": {"role": "assistant", "content": STATE.ollama_reply},
                                    "prompt_eval_count": 30, "eval_count": 6, "done": True})
        self._send(404, {"error": "nope"})


class FakeServer:
    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        STATE.reset()
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
