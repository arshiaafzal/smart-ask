import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "claude-smart-ask"


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def executable(path, source):
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    path.chmod(0o755)


class ClaudeSmartAskLauncherTests(unittest.TestCase):
    def test_runs_any_strategy_and_keeps_provider_key_out_of_claude(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            state_parent = temporary / "state"
            observed_path = temporary / "observed.json"
            metrics_path = temporary / "metrics.jsonl"
            trace_path = temporary / "traces.jsonl"
            secrets_path = temporary / "launcher.env"
            secrets_path.write_text(
                'OPENAI_API_KEY="test-provider-key"\n',
                encoding="utf-8",
            )
            port = unused_port()
            adapter = temporary / "fake-adapter"
            claude = temporary / "fake-claude"
            executable(adapter, """
                #!/usr/bin/env python3
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
                import json
                import os
                from pathlib import Path
                import sys

                config_path = Path(sys.argv[sys.argv.index("--config") + 1])
                config = json.loads(config_path.read_text(encoding="utf-8"))
                Path(os.environ["FAKE_CONFIG_COPY"]).write_text(json.dumps({
                    "config": config,
                    "openai_key": os.environ.get("OPENAI_API_KEY"),
                }), encoding="utf-8")

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/v1/models":
                            self.send_response(200)
                            self.send_header("content-type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps({"data": [{
                                "id": "claude-smart-ask-python-code-generation-codex-cascade"
                            }]}).encode())
                        else:
                            self.send_response(404)
                            self.end_headers()

                    def log_message(self, *_args):
                        pass

                listen = config["listen"]
                ThreadingHTTPServer(
                    (listen["host"], listen["port"]), Handler
                ).serve_forever()
            """)
            executable(claude, """
                #!/usr/bin/env python3
                import json
                import os
                import sys

                print(json.dumps({
                    "argv": sys.argv[1:],
                    "cwd": os.getcwd(),
                    "base_url": os.environ.get("ANTHROPIC_BASE_URL"),
                    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                    "openai_key": os.environ.get("OPENAI_API_KEY"),
                }))
            """)
            env = {
                **os.environ,
                "SMART_ASK_LAUNCHER_STATE_DIR": str(state_parent),
                "SMART_ASK_ADAPTER_BIN": str(adapter),
                "SMART_ASK_PYTHON": sys.executable,
                "SMART_ASK_AUTO_INSTALL": "0",
                "SMART_ASK_ADAPTER_PORT": str(port),
                "SMART_ASK_CLAUDE_CODE_TOKEN": "test-token",
                "SMART_ASK_START_ATTEMPTS": "40",
                "SMART_ASK_METRICS_PATH": str(metrics_path),
                "SMART_ASK_SECRETS_FILE": str(secrets_path),
                "CLAUDE_BIN": str(claude),
                "FAKE_CONFIG_COPY": str(observed_path),
            }
            env.pop("OPENAI_API_KEY", None)

            run = subprocess.run(
                [
                    str(LAUNCHER),
                    "--strategy",
                    "python-code-generation-codex-cascade",
                    "--trace-path",
                    str(trace_path),
                    "-p",
                    "hello",
                ],
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
                "claude-smart-ask-python-code-generation-codex-cascade",
                "-p",
                "hello",
            ])
            self.assertEqual(Path(payload["cwd"]).resolve(), temporary.resolve())
            self.assertEqual(payload["base_url"], f"http://127.0.0.1:{port}")
            self.assertEqual(payload["api_key"], "test-token")
            self.assertIsNone(payload["openai_key"])
            self.assertIn(
                f"metrics: {metrics_path.resolve()}",
                run.stderr,
            )
            self.assertIn(
                f"trace: {trace_path.resolve()} (contains conversation content)",
                run.stderr,
            )

            observed = json.loads(observed_path.read_text(encoding="utf-8"))
            self.assertEqual(observed["openai_key"], "test-provider-key")
            self.assertEqual(observed["config"]["strategies"], [
                "builtin:python-code-generation-codex-cascade",
            ])
            self.assertEqual(
                Path(observed["config"]["metrics"]["jsonl_path"]).resolve(),
                metrics_path.resolve(),
            )
            self.assertEqual(
                Path(
                    observed["config"]["metrics"]["trace_jsonl_path"]
                ).resolve(),
                trace_path.resolve(),
            )
            self.assertFalse(any(state_parent.glob("smart-ask-claude.*")))
            with socket.socket() as sock:
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)


if __name__ == "__main__":
    unittest.main()
