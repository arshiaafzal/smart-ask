import importlib.util
from pathlib import Path
import unittest

import smart_ask


class NoLegacyArchitectureTests(unittest.TestCase):
    def test_removed_modules_do_not_remain_as_shims(self):
        for module in (
            "smart_ask.application",
            "smart_ask.domain",
            "smart_ask.routing",
            "smart_ask.conversation.runtime",
            "smart_ask.conversation.executor",
            "smart_ask.methods.base",
            "smart_ask.methods.fixed",
            "smart_ask.methods.difficulty",
            "smart_ask.methods.cascade",
            "smart_ask.metrics.collector",
            "smart_ask.metrics.rollups",
            "smart_ask_claude_code",
        ):
            with self.subTest(module=module):
                self.assertIsNone(importlib.util.find_spec(module))

    def test_public_package_exposes_only_conversation_native_names(self):
        for name in (
            "Task",
            "Context",
            "SmartAsk",
            "SmartRouter",
            "ConversationRuntime",
            "ConversationRequest",
            "ExecutionRequest",
            "ModelResult",
            "RunStats",
            "StatsCollector",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(smart_ask, name))

    def test_old_provider_module_names_are_deleted(self):
        root = Path(__file__).parents[1] / "smart_ask" / "executors"
        self.assertEqual(list(root.glob("*_conversation.py")), [])

    def test_separate_claude_integration_package_is_deleted(self):
        root = Path(__file__).parents[1]
        self.assertFalse((root / "integrations" / "claude_code").exists())


if __name__ == "__main__":
    unittest.main()
