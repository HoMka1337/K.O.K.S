import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from agent.executor import Executor
from agent.memory import Memory
from agent.knowledge import Knowledge
from agent.planner import Planner
from core.tokenizer import BPETokenizer
from main import MyCodex
from tools.web_tools import WebTools


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.codex = MyCodex()

    def test_greeting(self):
        self.assertIsNotNone(self.codex.execute_action("привет"))

    def test_arithmetic(self):
        self.assertEqual(
            self.codex.execute_action("посчитай 2 + 2"),
            "2.0 + 2.0 = 4.0",
        )

    def test_dangerous_request_is_detected(self):
        self.assertTrue(self.codex.is_dangerous_request("удали папку temp"))
        self.assertFalse(self.codex.is_dangerous_request("покажи список файлов"))

    def test_text_file_command_requires_a_filename(self):
        response = self.codex.execute_action("создай текстовый документ")
        self.assertEqual(response, "Укажи имя файла, например: создай файл notes.txt")

    def test_system_commands(self):
        time_response = cast(str, self.codex.execute_action("который час"))
        date_response = cast(str, self.codex.execute_action("дата"))
        system_response = cast(str, self.codex.execute_action("информация о системе"))
        self.assertIsInstance(time_response, str)
        self.assertIsInstance(date_response, str)
        self.assertIsInstance(system_response, str)
        self.assertRegex(time_response, r"^Сейчас \d{2}:\d{2}:\d{2}$")
        self.assertRegex(date_response, r"^Сегодня \d{2}\.\d{2}\.\d{4}$")
        self.assertIn("OS:", system_response)

    def test_web_search_requires_a_query(self):
        self.assertEqual(WebTools.search(""), "Укажи, что искать в интернете")

    def test_remember_url(self):
        response = self.codex.execute_action("запомни это https://example.com/article")
        self.assertEqual(response, "Запомнил ссылку: https://example.com/article")

    def test_nested_tool_action(self):
        planner = Planner()
        executor = Executor()
        executor.register_tool("echo", lambda value: value, "")
        action = planner.parse_action(
            json.dumps({"tool": "echo", "args": {"value": "ok"}})
        )
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(executor.execute(action["tool"], action["args"]), "ok")

    def test_memory_accepts_path(self):
        memory = Memory(Path("memory-test.json"))
        self.assertEqual(memory.memory_path, Path("memory-test.json"))

    def test_knowledge_search(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Knowledge(Path(directory) / "knowledge.json")
            knowledge.add_text("Проект использует локальную Transformer модель", "notes")
            self.assertIn("Transformer", knowledge.search("модель"))

    def test_bpe_merges_are_used(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["aaaa"], vocab_size=20)
        encoded = tokenizer.encode("aaaa")
        self.assertLess(len(encoded), 6)
        self.assertEqual(tokenizer.decode(encoded), "aaaa")

    def test_multi_step_agent_flow(self):
        responses = iter([
            '{"tool": "echo", "args": {"value": "step one"}}',
            '{"done": true, "result": "Задача завершена"}',
        ])

        class FakeInference:
            def generate(self, *args, **kwargs):
                return next(responses)

        self.codex.model = object()
        self.codex.inference = FakeInference()
        self.codex.executor.register_tool("echo", lambda value: value, "")
        self.assertEqual(self.codex.run("сделай два шага"), "Задача завершена")


if __name__ == "__main__":
    unittest.main()
