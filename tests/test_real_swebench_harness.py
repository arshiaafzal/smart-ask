import unittest

from benchmark.real_swebench import _agent_prompt, _cache_system_prompt


class RealSweBenchHarnessTests(unittest.TestCase):
    def test_cache_namespace_is_separate_from_task_instructions(self):
        task = {"problem_statement": "Numbers are mistaken for docstrings."}

        prompt = _agent_prompt(task)
        cache_prompt = _cache_system_prompt("run-abc")

        self.assertIn("Numbers are mistaken for docstrings.", prompt)
        self.assertIn("Use .venv/bin/python -m pytest", prompt)
        self.assertNotIn("cache namespace", prompt)
        self.assertIn("Evaluation cache namespace: run-abc", cache_prompt)
        self.assertIn("has no bearing on the task", cache_prompt)


if __name__ == "__main__":
    unittest.main()
