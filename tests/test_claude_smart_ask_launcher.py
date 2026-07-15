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
            trace_directory = temporary / "trace-session"
            secrets_path = temporary / "launcher.env"
            secrets_path.write_text(
                'OPENAI_API_KEY="test-provider-key"\n',
                encoding="utf-8",
            )
            port = unused_port()
            gateway = temporary / "fake-gateway"
            claude = temporary / "fake-claude"
            executable(gateway, """
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
                                "id": "smart-ask-python-code-generation-codex-cascade"
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
                "SMART_ASK_GATEWAY_BIN": str(gateway),
                "SMART_ASK_PYTHON": sys.executable,
                "SMART_ASK_AUTO_INSTALL": "0",
                "SMART_ASK_GATEWAY_PORT": str(port),
                "SMART_ASK_GATEWAY_TOKEN": "test-token",
                "SMART_ASK_START_ATTEMPTS": "40",
                "SMART_ASK_METRICS_PATH": str(metrics_path),
                "SMART_ASK_SECRETS_FILE": str(secrets_path),
                "CLAUDE_BIN": str(claude),
                "FAKE_CONFIG_COPY": str(observed_path),
            }
            env.pop("OPENAI_API_KEY", None)
            env.pop("SMART_ASK_TRACE_DIR", None)

            run = subprocess.run(
                [
                    str(LAUNCHER),
                    "--strategy",
                    "python-code-generation-codex-cascade",
                    "--trace-dir",
                    str(trace_directory),
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
                "smart-ask-python-code-generation-codex-cascade",
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
                "trace directory: "
                f"{trace_directory.resolve()} (contains conversation content)",
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
                    observed["config"]["metrics"]["trace_directory"]
                ).resolve(),
                trace_directory.resolve(),
            )
            self.assertFalse(any(state_parent.glob("smart-ask-claude.*")))
            with socket.socket() as sock:
                self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)

            default_env = dict(env)
            default_env.pop("SMART_ASK_METRICS_PATH")
            traced = subprocess.run(
                [
                    str(LAUNCHER),
                    "--strategy",
                    "python-code-generation-codex-cascade",
                    "--trace",
                    "-p",
                    "hello",
                ],
                cwd=temporary,
                env=default_env,
                text=True,
                capture_output=True,
                check=True,
                timeout=20,
            )
            traced_config = json.loads(
                observed_path.read_text(encoding="utf-8")
            )["config"]
            generated_trace_directory = Path(
                traced_config["metrics"]["trace_directory"]
            )
            self.assertEqual(
                generated_trace_directory.parent,
                ROOT / ".smart-ask" / "claude-code" / "traces",
            )
            self.assertRegex(
                generated_trace_directory.name,
                r"^\d{8}T\d{6}Z-[0-9a-f]{8}$",
            )
            self.assertIn(
                f"trace directory: {generated_trace_directory}",
                traced.stderr,
            )
            self.assertEqual(
                Path(traced_config["metrics"]["jsonl_path"]),
                ROOT / ".smart-ask" / "claude-code" / "metrics.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
