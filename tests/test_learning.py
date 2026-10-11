import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from config import Config
from core.learning import LearningStore, prepare_training, question_key
from core.model import CodexModel
from core.tokenizer import BPETokenizer
from core.trainer import CodexTrainer
from life.journal import Journal
from tools.security import SecurityManager


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(Config, "DATA_DIR", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_pending_never_trains_and_correction_wins(self):
        store = LearningStore()
        store.add("question", "bad", "web")
        self.assertEqual(store.batch(20), [])
        store.add("question", "first", "human", True)
        store.add("question", "corrected", "human", True)
        self.assertEqual([r["answer"] for r in store.verified()], ["corrected"])
        store.add("question", "untrusted", "web")
        self.assertEqual(store.verified()[0]["answer"], "corrected")

    def test_recent_data_and_replay(self):
        store = LearningStore()
        for i in range(20):
            store.add(f"old{i}", "answer", "human", True)
        store.mark_trained([f"old{i}\nanswer" for i in range(20)])
        for i in range(6):
            store.add(f"new{i}", "answer", "human", True)
        rows = store.batch(10, require_new=True)
        self.assertEqual(sum(r["trained"] == 0 for r in rows), 2)
        self.assertEqual(sum(r["trained"] > 0 for r in rows), 8)
        self.assertIn("new5", [r["question"] for r in rows])

    def test_no_new_examples_no_auto_training(self):
        store = LearningStore()
        for i in range(5):
            store.add(f"q{i}", "answer", "human", True)
        store.mark_trained([f"q{i}\nanswer" for i in range(5)])
        self.assertEqual(store.batch(20, require_new=True), [])

    def test_fixed_holdout_excludes_corrected_questions(self):
        texts = [f"question{i}\nanswer" for i in range(20)]
        training, validation = prepare_training(texts, 2000)
        first = (self.root / "validation.json").read_bytes()
        corrected = validation[0].split("\n")[0] + "\nnew answer"
        training, second = prepare_training(texts + [corrected, "new\nanswer"], 2000)
        self.assertEqual(first, (self.root / "validation.json").read_bytes())
        self.assertEqual(validation, second)
        self.assertNotIn(question_key(corrected), {question_key(t) for t in training})

    def test_not_enough_data(self):
        with self.assertRaises(ValueError):
            prepare_training(["q\na"], 2000)

    def test_token_boundaries(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["question answer"], 40)
        self.assertNotEqual(tokenizer.encode("question\nanswer"), tokenizer.encode("question answer"))
        self.assertEqual(tokenizer.encode_prompt("question")[-1], tokenizer.vocab["<SEP>"])
        self.assertNotIn(tokenizer.vocab["<EOS>"], tokenizer.encode_prompt("question"))

    def test_legacy_tokenizer_preserves_format(self):
        tokenizer = BPETokenizer()
        tokenizer.train(["question answer"], 40)
        path = self.root / "tokenizer.json"
        tokenizer.save(path)
        obj = json.loads(path.read_text()); obj.pop("format_version")
        path.write_text(json.dumps(obj))
        tokenizer.load(path)
        self.assertEqual(tokenizer.format_version, 1)
        self.assertEqual(tokenizer.encode("question\nanswer"), tokenizer.encode("question answer"))
        self.assertNotIn(tokenizer.vocab["<EOS>"], tokenizer.encode_prompt("question"))

    def test_lock_survives_restart(self):
        path = self.root / "security.json"
        path.write_text(json.dumps({"locked": True}))
        with patch.object(Config, "SECURITY_STATE_FILE", path):
            self.assertTrue(SecurityManager().locked)

    def test_journal_bounded_during_session(self):
        with patch.object(Config, "LIFE_JOURNAL_KEEP", 2):
            journal = Journal(self.root / "journal.jsonl")
            for i in range(5):
                journal.add("thought", str(i))
            self.assertEqual([r["content"] for r in journal.entries], ["3", "4"])
            self.assertEqual(len(journal.path.read_text().splitlines()), 2)

    def test_python_execution_disabled(self):
        self.assertFalse(Config.can_execute_code("open('C:/outside.txt', 'w').write('test')"))

    def trainer(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        tokenizer = BPETokenizer()
        tokenizer.train(["question answer another reply"], 40)
        model = CodexModel(len(tokenizer), d_model=8, n_heads=2, n_layers=1,
                           d_ff=16, max_len=32, dropout=0)
        return CodexTrainer(model, tokenizer, "cpu")

    def test_rejected_candidate_restores_weights_and_file(self):
        trainer = self.trainer()
        path = self.root / "model.pt"
        trainer.save(str(path))
        before_file = path.read_bytes()
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with patch.object(trainer, "evaluate", side_effect=[1.0, 2.0]):
            accepted = trainer.train(["question\nanswer"], epochs=1, max_len=24,
                                     validation_texts=["another\nreply"], save_path=str(path))
        self.assertFalse(accepted)
        self.assertEqual(path.read_bytes(), before_file)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items()))

    def test_best_epoch_restored_and_previous_checkpoint_kept(self):
        trainer = self.trainer()
        path = self.root / "model.pt"
        trainer.save(str(path)); old = path.read_bytes()
        best = {}
        def evaluate(loader):
            evaluate.calls += 1
            if evaluate.calls == 2:
                best.update({k: v.clone() for k, v in trainer.model.state_dict().items()})
            return [3.0, 2.0, 4.0][evaluate.calls - 1]
        evaluate.calls = 0
        with patch.object(trainer, "evaluate", side_effect=evaluate):
            self.assertTrue(trainer.train(["question\nanswer"], epochs=2, max_len=24,
                            validation_texts=["another\nreply"], save_path=str(path)))
        self.assertEqual(Path(str(path) + ".previous").read_bytes(), old)
        self.assertTrue(all(torch.equal(v, best[k]) for k, v in trainer.model.state_dict().items()))
        saved = torch.load(path, weights_only=True)["model_state_dict"]
        self.assertTrue(all(torch.equal(v, saved[k]) for k, v in trainer.model.state_dict().items()))

    def test_training_exception_restores_model(self):
        trainer = self.trainer()
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with patch.object(trainer, "evaluate", side_effect=[1.0, RuntimeError("test")]):
            with self.assertRaises(RuntimeError):
                trainer.train(["question\nanswer"], epochs=1, max_len=24,
                              validation_texts=["another\nreply"], save_path=str(self.root / "m.pt"))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items()))
        self.assertFalse((self.root / "m.pt").exists())

    def test_real_training_evaluation(self):
        trainer = self.trainer()
        result = trainer.train(["question\nanswer"], epochs=1, max_len=24,
                               validation_texts=["another\nreply"], save_path=str(self.root / "m.pt"))
        self.assertIsInstance(result, bool)
        self.assertEqual((self.root / "m.pt").exists(), result)

    def test_full_context_can_generate(self):
        trainer = self.trainer()
        trainer.model.max_len = 4
        with patch.object(trainer.model, "_clean_response", side_effect=lambda text, prompt: text):
            with patch("torch.multinomial", return_value=torch.tensor([trainer.tokenizer.vocab["a"]])):
                self.assertEqual(trainer.model.generate(trainer.tokenizer, "question " * 10, max_new_tokens=2), "aa")

    def test_checkpoint_rejects_wrong_tokenizer(self):
        trainer = self.trainer()
        path = self.root / "model.pt"
        trainer.save(str(path))
        trainer.tokenizer.format_version = 1
        with self.assertRaises(ValueError):
            trainer.load(str(path))

    def test_failed_save_keeps_checkpoint_and_restores_model(self):
        trainer = self.trainer()
        path = self.root / "model.pt"
        trainer.save(str(path)); previous = path.read_bytes()
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with patch.object(trainer, "evaluate", side_effect=[3.0, 2.0]):
            with patch.object(trainer, "_save_checkpoint", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    trainer.train(["question\nanswer"], epochs=1, max_len=24,
                                  validation_texts=["another\nreply"], save_path=str(path))
        self.assertEqual(path.read_bytes(), previous)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items()))

    def test_web_candidates_do_not_enter_dialogs(self):
        from life.autolearn import AutoLearner
        from types import SimpleNamespace
        learner = AutoLearner(SimpleNamespace(), Journal(self.root / "journal.jsonl"))
        with patch.object(Config, "can_write", return_value=True):
            with patch.object(learner, "extract_answer_from_page", return_value="Relevant factual answer " * 5):
                self.assertTrue(learner.learn_from_page("valid question", "page", source="https://example.com"))
        self.assertFalse((self.root / "dialogs.txt").exists())
        self.assertEqual(learner.learning.verified(), [])
        self.assertEqual(learner.learning.records()[0]["source"], "https://example.com")

    def test_mask_is_applied_when_supplied(self):
        from core.model import CausalSelfAttention
        attention = CausalSelfAttention(8, 2, dropout=0).eval()
        inputs = torch.randn(1, 3, 8)
        changed = inputs.clone(); changed[:, 2, :] += 100
        mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
        self.assertTrue(torch.allclose(attention(inputs, mask)[:, :2], attention(changed, mask)[:, :2]))

    def test_structured_action_survives_response_cleanup(self):
        trainer = self.trainer()
        response = '{"tool": "read_file", "args": {"path": "notes.txt"}}'
        self.assertEqual(trainer.model._clean_response(response, ""), response)

    def test_application_training_tracks_only_training_examples(self):
        from main import MyCodex
        from types import SimpleNamespace
        trainer = self.trainer()
        store = LearningStore()
        for i in range(12):
            store.add(f"question {i}", "answer", "human", True)
        app = SimpleNamespace(learning=store, tokenizer=trainer.tokenizer, trainer=trainer)
        path = self.root / "app-model.pt"
        with patch.object(Config, "MODEL_FILE", path):
            with patch.object(Config, "TEACHER_DIALOGS_FILE", self.root / "teacher.txt"):
                with patch.object(trainer, "evaluate", side_effect=[3.0, 2.0]):
                    self.assertTrue(MyCodex.train_from_dialogs(app, epochs=1))
        held_out = {question_key(t) for t in json.loads((self.root / "validation.json").read_text())}
        for row in store.verified():
            self.assertEqual(row["trained"], 0 if question_key(row["question"]) in held_out else 1)
        self.assertEqual(store.batch(2000, require_new=True), [])

    def test_early_stopping(self):
        trainer = self.trainer()
        with patch.object(Config, "EARLY_STOPPING_PATIENCE", 2):
            with patch.object(trainer, "evaluate", side_effect=[3.0, 2.0, 2.1, 2.2]) as evaluate:
                self.assertTrue(trainer.train(["question\nanswer"], epochs=10, max_len=24,
                    validation_texts=["another\nreply"], save_path=str(self.root / "m.pt")))
                self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(trainer.last_training_metrics["best_epoch"], 1)
        self.assertEqual(len(trainer.last_training_metrics["epochs"]), 3)

    def test_prompt_style_survives_checkpoint(self):
        trainer = self.trainer()
        path = self.root / "model.pt"
        trainer.model.prompt_style = "plain_dialog"
        trainer.save(str(path))
        trainer.model.prompt_style = "agent"
        trainer.load(str(path))
        self.assertEqual(trainer.model.prompt_style, "plain_dialog")

    def test_plain_dialog_model_gets_training_format(self):
        from main import MyCodex
        from types import SimpleNamespace
        from unittest.mock import Mock
        app = MyCodex()
        app.model = SimpleNamespace(prompt_style="plain_dialog")
        app.inference = Mock()
        app.inference.generate.return_value = "Это достаточно длинный ответ для проверки"
        app.run("уникальный вопрос о формате запроса")
        self.assertEqual(app.inference.generate.call_args.args[0], "уникальный вопрос о формате запроса")

    def test_unready_model_does_not_generate_in_chat(self):
        from main import MyCodex
        from types import SimpleNamespace
        from unittest.mock import Mock
        app = MyCodex()
        app.model = SimpleNamespace(generation_ready=False)
        app.inference = Mock()
        response = app.run("квазиспектральный гиперкуб щщщщ")
        app.inference.generate.assert_not_called()
        self.assertIn("Не знаю", response)

    def test_generation_readiness_survives_checkpoint(self):
        trainer = self.trainer()
        path = self.root / "model.pt"
        trainer.model.generation_ready = False
        trainer.save(str(path))
        trainer.model.generation_ready = True
        trainer.load(str(path))
        self.assertFalse(trainer.model.generation_ready)
