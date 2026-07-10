import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "claude-local-qwen"


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def executable(path, source):
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(0o755)


class ClaudeLocalQwenLauncherTests(unittest.TestCase):
    def test_starts_reuses_and_stops_background_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            state = temporary / "state"
            ollama_port = unused_port()
            adapter_port = unused_port()
            ollama = temporary / "fake-ollama"
            adapter = temporary / "fake-adapter"
            claude = temporary / "fake-claude"
            executable(ollama, f"""
                #!/usr/bin/env python3
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/api/version":
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(b'{{"version":"test"}}')
                        else:
                            self.send_response(404)
                            self.end_headers()

                    def log_message(self, *_args):
                        pass

                ThreadingHTTPServer(("127.0.0.1", {ollama_port}), Handler).serve_forever()
            """)
            executable(adapter, f"""
                #!/usr/bin/env python3
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/healthz":
                            self.send_response(200)
                            self.end_headers()
                        elif self.path == "/v1/models":
                            self.send_response(200)
                            self.end_headers()
                            self.wfile.write(
                                b'{{"data":[{{"id":"claude-smart-ask-local-qwen"}}]}}'
                            )
                        else:
                            self.send_response(404)
                            self.end_headers()

                    def log_message(self, *_args):
                        pass

                ThreadingHTTPServer(("127.0.0.1", {adapter_port}), Handler).serve_forever()
            """)
            executable(claude, """
                #!/usr/bin/env python3
                import json
                import os
                import sys

                print(json.dumps({
                    "argv": sys.argv[1:],
                    "base_url": os.environ.get("ANTHROPIC_BASE_URL"),
                    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                    "openrouter_key": os.environ.get("OPENROUTER_API_KEY"),
                }))
            """)
            env = {
                **os.environ,
                "SMART_ASK_LAUNCHER_STATE_DIR": str(state),
                "SMART_ASK_OLLAMA_BIN": str(ollama),
                "SMART_ASK_ADAPTER_BIN": str(adapter),
                "CLAUDE_BIN": str(claude),
                "SMART_ASK_OLLAMA_URL": f"http://127.0.0.1:{ollama_port}",
                "SMART_ASK_ADAPTER_URL": f"http://127.0.0.1:{adapter_port}",
                "SMART_ASK_CLAUDE_CODE_TOKEN": "test-token",
                "SMART_ASK_START_ATTEMPTS": "40",
                "OPENROUTER_API_KEY": "must-not-reach-children",
            }

            try:
                run = subprocess.run(
                    [str(LAUNCHER), "-p", "hello"],
                    cwd=temporary,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=20,
                )
                payload = json.loads(run.stdout.strip().splitlines()[-1])
                self.assertEqual(payload["argv"], [
                    "--model",
                    "claude-smart-ask-local-qwen",
                    "-p",
                    "hello",
                ])
                self.assertEqual(
                    payload["base_url"],
                    f"http://127.0.0.1:{adapter_port}",
                )
                self.assertEqual(payload["api_key"], "test-token")
                self.assertIsNone(payload["openrouter_key"])

                status = subprocess.run(
                    [str(LAUNCHER), "status"],
                    cwd=temporary,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                self.assertIn("Ollama: ready (started here", status.stdout)
                self.assertIn("Adapter: ready (started here", status.stdout)

                stopped = subprocess.run(
                    [str(LAUNCHER), "stop"],
                    cwd=temporary,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=15,
                )
                self.assertIn("Adapter: stopped", stopped.stdout)
                self.assertIn("Ollama: stopped", stopped.stdout)
            finally:
                for name in ("adapter.pid", "ollama.pid"):
                    path = state / name
                    if not path.exists():
                        continue
                    try:
                        os.kill(int(path.read_text(encoding="utf-8")), 15)
                    except (OSError, ValueError):
                        pass


if __name__ == "__main__":
    unittest.main()
