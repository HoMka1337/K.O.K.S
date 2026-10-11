#!/usr/bin/env python3
import os
import sys
import json
import re
import random
import subprocess
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.model import CodexModel
from core.tokenizer import BPETokenizer
from core.trainer import CodexTrainer
from core.learning import LearningStore, prepare_training, question_key
from core.inference import CodexInference
from core.rag import RagEngine

from tools.file_tools import FileTools
from tools.system_tools import SystemTools
from tools.code_tools import CodeTools
from tools.web_tools import WebTools
from tools.security import get_security
from tools.corpus import build_corpus_jsonl

from agent.memory import Memory
from agent.knowledge import Knowledge
from agent.planner import Planner
from agent.executor import Executor
from agent.dialogs import DialogStore
from agent.corpus_index import CorpusIndex
from agent.semantics import answer_looks_meaningful, clean_answer, detect_question_type


class MyCodex:
    def __init__(self):
        Config.init()
        Config.save_snapshot()
        Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=Config.LOG_FILE,
            level=logging.INFO,
            format=Config.LOG_FORMAT,
            datefmt=Config.LOG_DATE_FORMAT,
            encoding="utf-8",
        )
        self.logger = logging.getLogger("koks")
        self.security = get_security()
        self.tokenizer = BPETokenizer()
        self.model = None
        self.trainer = None
        self.inference = None
        self.executor = Executor()
        self.memory = Memory(Config.MEMORY_FILE)
        self.knowledge = Knowledge(Config.KNOWLEDGE_FILE)
        self.dialogs = DialogStore()
        self.corpus_index = CorpusIndex()
        self.rag = RagEngine(self.corpus_index)
        self._register_tools()
        self.planner = Planner()
        self.last_question = None
        self.learned_commands = []
        self.teaching_mode = False
        self.learning = LearningStore()
        self.learned_answers = self._load_learned_answers()
        self.load_learned_commands()
        print("=" * 60)
        print("KoKs")
        print("=" * 60)
        print(Config.security_report())
        print("=" * 60)
        if self.rag.loaded:
            stats = self.rag.stats()
            print(f"RAG: индекс загружен, документов: {stats.get('n_docs', 0)}")
        else:
            print("RAG: индекс не построен")
        print("=" * 60)

    def load_learned_commands(self):
        cmd_file = Config.LEARNED_COMMANDS_FILE
        if cmd_file.exists():
            try:
                self.learned_commands = [
                    line.strip()
                    for line in cmd_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:
                self.learned_commands = []

    def save_learned_commands(self):
        Config.LEARNED_COMMANDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            Config.LEARNED_COMMANDS_FILE.write_text(
                "\n".join(self.learned_commands), encoding="utf-8"
            )
        except OSError:
            pass

    def _load_learned_answers(self):
        answers = {}
        for file_path in (Config.DIALOGS_FILE, Config.TEACHER_DIALOGS_FILE):
            if not file_path.exists():
                continue
            try:
                blocks = file_path.read_text(encoding="utf-8").split("\n\n")
                for block in blocks:
                    lines = [l.strip() for l in block.splitlines() if l.strip()]
                    if len(lines) >= 2:
                        answers[self._normalize_question(lines[0])] = lines[1]
            except (OSError, UnicodeError):
                continue
        for row in self.learning.verified():
            answers[self._normalize_question(row["question"])] = row["answer"]
        return answers

    @staticmethod
    def _normalize_question(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE).strip()

    def _get_learned_answer(self, question: str) -> Optional[str]:
        normalized = self._normalize_question(question)
        if normalized in self.learned_answers:
            answer = self.learned_answers[normalized]
            if answer_looks_meaningful(answer):
                return clean_answer(answer)
        return None

    def _semantic_answer(self, question: str) -> Optional[str]:
        answer = self.dialogs.find(question, min_score=0.5)
        if answer:
            return answer
        answer = self.knowledge.search(question, limit=1)
        if answer:
            return answer
        return None

    def _register_tools(self):
        tools = [
            ("read_file", FileTools.read_file, "чтение файла внутри проекта"),
            ("write_file", FileTools.write_file, "запись файла внутри проекта"),
            ("append_file", FileTools.append_file, "дозапись в файл внутри проекта"),
            ("list_files", FileTools.list_files, "список файлов"),
            ("delete_file", FileTools.delete_file, "удаление только в разрешённых папках"),
            ("copy_file", FileTools.copy_file, "копирование внутри проекта"),
            ("move_file", FileTools.move_file, "перемещение внутри проекта"),
            ("execute_command", SystemTools.execute_command, "только из белого списка"),
            ("create_folder", FileTools.create_folder, "создание папки внутри проекта"),
            ("execute_python", CodeTools.execute_python, "запуск кода в песочнице"),
            ("analyze_python", CodeTools.analyze_python, "статический анализ"),
            ("web_search", WebTools.search, "поиск в интернете"),
            ("search_knowledge", self.knowledge.search, "поиск по знаниям"),
        ]
        for name, func, desc in tools:
            self.executor.register_tool(name, func, desc)

    def create_model(
        self,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_len: Optional[int] = None,
    ):
        if vocab_size is None:
            vocab_size = len(self.tokenizer) if self.tokenizer else Config.VOCAB_SIZE
        if vocab_size < 10:
            vocab_size = Config.VOCAB_SIZE
        d_model = d_model or Config.D_MODEL
        n_heads = n_heads or Config.N_HEADS
        n_layers = n_layers or Config.N_LAYERS
        d_ff = d_ff or Config.D_FF
        max_len = max_len or Config.MAX_LEN
        if d_model % n_heads != 0:
            for candidate in (8, 4, 2, 1):
                if d_model % candidate == 0:
                    n_heads = candidate
                    break
        print(
            f"Создание модели: vocab={vocab_size}, d_model={d_model}, "
            f"n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}, max_len={max_len}"
        )
        self.model = CodexModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=Config.DROPOUT,
        )
        self.trainer = CodexTrainer(self.model, self.tokenizer, Config.DEVICE)
        self.inference = CodexInference(self.model, self.tokenizer, Config.DEVICE)
        print("Модель создана")
        return self.model

    def train_from_dialogs(self, epochs=10, texts=None):
        if texts is None:
            # Only explicit human corrections and teacher examples are trusted.
            if Config.TEACHER_DIALOGS_FILE.exists():
                for block in Config.TEACHER_DIALOGS_FILE.read_text(encoding="utf-8").split("\n\n"):
                    if "\n" in block.strip():
                        q, a = block.strip().split("\n", 1)
                        if not any(question_key(row["question"]) == question_key(q) for row in self.learning.verified()):
                            self.learning.add(q, a, "teacher_dialogs.txt", verified=True)
            rows = self.learning.batch(Config.LIFE_MAX_TRAIN_TEXTS)
            texts = [row["question"] + "\n" + row["answer"] for row in rows]
        if not texts:
            print("Нет проверенных примеров для обучения")
            return False
        try:
            texts, validation = prepare_training(texts, Config.LIFE_MAX_TRAIN_TEXTS)
        except ValueError as error:
            print(f"Обучение отложено: {error}")
            return False
        print(f"\n[Обучение] Найдено примеров: {len(texts)}")
        if not self.tokenizer.vocab:
            self.tokenizer.train(texts, min(Config.VOCAB_SIZE, 5000))
            self.tokenizer.save(str(Config.TOKENIZER_FILE))
            vocab_size = self.tokenizer.vocab_size
            self.create_model(vocab_size)
        if self.trainer is None:
            self.create_model(len(self.tokenizer))
        assert self.trainer is not None
        accepted = self.trainer.train(
            texts=texts,
            epochs=epochs,
            batch_size=min(Config.TRAIN_BATCH_SIZE, max(1, len(texts))),
            lr=Config.LEARNING_RATE,
            max_len=min(Config.TRAIN_MAX_LEN, Config.MAX_LEN),
            save_path=str(Config.MODEL_FILE),
            validation_texts=validation,
        )
        if accepted:
            self.learning.mark_trained(texts)
        print("[Обучение завершено]\n")
        return accepted

    def train_from_corpus(self, epochs=1, max_texts=None):
        corpus_path = Config.DATA_DIR / "corpus.jsonl"
        if not corpus_path.exists():
            print(f"Корпус не найден: {corpus_path}")
            return
        texts = []
        limit = max_texts or Config.LIFE_MAX_TRAIN_TEXTS
        sampler = random.Random(42)
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if len(text) < 60:
                    continue
                texts.append(text)
                if len(texts) > limit:
                    # Keep recent examples plus a bounded random sample of older ones.
                    del texts[sampler.randrange(max(1, len(texts) - max(1, limit // 5)))]
        if not texts:
            print("В корпусе нет подходящих текстов")
            return
        self.train_from_dialogs(epochs=epochs, texts=texts)

    def load(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        model_path = model_path or str(Config.MODEL_FILE)
        tokenizer_path = tokenizer_path or str(Config.TOKENIZER_FILE)
        if not Path(model_path).exists():
            print("Модель не найдена. Создаю новую...")
            self.create_model()
            return True
        if not Path(tokenizer_path).exists():
            print("Нет токенизатора для сохранённой модели; загрузка отменена")
            return False
        if Path(tokenizer_path).exists():
            try:
                self.tokenizer.load(tokenizer_path)
            except Exception as e:
                print(f"Ошибка токенизатора: {e}")
                return False
        from core.trainer import read_checkpoint_shape
        shape = read_checkpoint_shape(model_path)
        if shape is None:
            print("Не удалось прочитать checkpoint; загрузка отменена")
            return False
        vocab_size = shape.get("vocab_size") or (
            len(self.tokenizer) if self.tokenizer else Config.VOCAB_SIZE
        )
        if vocab_size < 10:
            vocab_size = Config.VOCAB_SIZE
        self.create_model(
            vocab_size=vocab_size,
            d_model=shape.get("d_model"),
            n_heads=shape.get("n_heads"),
            n_layers=shape.get("n_layers"),
            d_ff=shape.get("d_ff"),
            max_len=shape.get("max_len"),
        )
        try:
            assert self.trainer is not None
            self.trainer.load(model_path)
            print("Модель загружена")
            return True
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            self.model = self.trainer = self.inference = None
            return False

    def _confirm_dangerous(self, action: str) -> bool:
        if not Config.CONFIRM_DANGEROUS:
            return True
        print(f"\n!!! ОПАСНОЕ ДЕЙСТВИЕ: {action}")
        if Config.REQUIRE_CONFIRMATION_CODE:
            code = self.security.generate_confirmation_code(action)
            print(f"Введи код подтверждения: {code}")
            entered = input("Код: ").strip()
            if not self.security.verify_confirmation_code(entered):
                Config._log_security(f"confirmation failed: {action[:80]}", "WARN")
                return False
            return True
        answer = input("Подтвердить? (да/нет): ").strip().lower()
        return answer in {"да", "д", "yes", "y"}

    def execute_action(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        greeting = text_lower.strip().strip("!?.,:;")
        if greeting in {"привет", "здравствуй", "здравствуйте", "ку", "хай", "салют"}:
            return random.choice([
                "Привет! Рад тебя видеть.",
                "Салют! Как дела?",
                "Ку! Чем займёмся?",
                "Привет! Что будем делать?",
            ])
        if text_lower.startswith("поищи в интернете") or text_lower.startswith("найди в интернете"):
            prefix = (
                "поищи в интернете"
                if text_lower.startswith("поищи в интернете")
                else "найди в интернете"
            )
            query = text[len(prefix):].strip()
            if not Config.ALLOW_NETWORK:
                return "Сеть отключена"
            return WebTools.search(query)
        if text_lower in {"который час", "сколько времени", "время"}:
            return datetime.now().strftime("Сейчас %H:%M:%S")
        if text_lower in {"какая сегодня дата", "какое сегодня число", "дата"}:
            return datetime.now().strftime("Сегодня %d.%m.%Y")
        if text_lower in {"информация о системе", "системная информация", "характеристики компьютера"}:
            return SystemTools.get_system_info()
        if text_lower.startswith("запомни это"):
            url_match = re.search(r"https?://\S+", text)
            if not url_match:
                return "Пришли ссылку после команды: запомни это <ссылка>"
            url = url_match.group(0).rstrip(".,);]")
            self.knowledge.add_url(url)
            return f"Запомнил ссылку: {url}"
        if text_lower.startswith("запомни файл"):
            file_path = text[len("запомни файл"):].strip()
            if not file_path:
                return "Укажи имя файла"
            if not Config.can_read(file_path):
                return f"Чтение запрещено: {file_path}"
            return self.knowledge.import_file(file_path)
        if "создай файл" in text_lower or "создай текстовый документ" in text_lower:
            match = re.search(
                r"(?:создай файл|создай текстовый документ)\s+(\S+)(?:\s+(?:с текстом|текст:)\s*(.*))?$",
                text,
                re.IGNORECASE,
            )
            if not match:
                return "Укажи имя файла, например: создай файл notes.txt"
            file_path = match.group(1)
            content = match.group(2) or ""
            if not Config.can_write(file_path):
                return f"Запись запрещена: {file_path}"
            if Config.CONFIRM_DANGEROUS and not self._confirm_dangerous(f"создать файл {file_path}"):
                return "Действие отменено"
            return FileTools.write_file(file_path, content)
        if text_lower.startswith("покажи файлы") or text_lower.startswith("список файлов"):
            folder = text.split(maxsplit=2)[-1] if len(text.split()) > 2 else "."
            return FileTools.list_files(folder)
        if text_lower.startswith("прочитай файл"):
            file_path = text[len("прочитай файл"):].strip()
            if not file_path:
                return "Укажи имя файла"
            return FileTools.read_file(file_path)
        if text_lower.startswith("найди файл"):
            file_name = text[len("найди файл"):].strip()
            if not file_name:
                return "Укажи имя файла для поиска"
            try:
                matches = [
                    str(path)
                    for path in Config.BASE_DIR.rglob(file_name)
                    if Config.can_read(path)
                ]
            except Exception as e:
                return f"Ошибка поиска: {e}"
            return "\n".join(matches) if matches else f"Файл не найден: {file_name}"
        if text_lower.startswith("открой файл"):
            return "Открытие файлов ОС отключено политикой безопасности"
        if "открой папку" in text_lower:
            return "Открытие папок ОС отключено политикой безопасности"
        if "создай папку" in text_lower or "сделай папку" in text_lower:
            match = re.search(r"папку\s+(\S+)", text_lower)
            folder_name = match.group(1) if match else "new_folder"
            if not Config.can_write(folder_name):
                return f"Создание папки запрещено: {folder_name}"
            if Config.CONFIRM_DANGEROUS and not self._confirm_dangerous(f"создать папку {folder_name}"):
                return "Действие отменено"
            return FileTools.create_folder(folder_name)
        if "удали папку" in text_lower:
            return "Удаление папок отключено политикой безопасности"
        if text_lower.startswith("удали файл"):
            return "Удаление файлов отключено политикой безопасности"
        if "выполни команду" in text_lower:
            return "Выполнение команд отключено политикой безопасности"
        if "посчитай" in text_lower or "реши" in text_lower or "сколько будет" in text_lower:
            match = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", text_lower)
            if match:
                a = float(match.group(1))
                op = match.group(2)
                b = float(match.group(3))
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                elif op == "/":
                    if b != 0:
                        result = a / b
                    else:
                        return "На ноль делить нельзя"
                else:
                    return "Неизвестная операция"
                return f"{a} {op} {b} = {result}"
        return None

    def _remember_exchange(self, user_input: str, response: str) -> None:
        self.memory.add({"role": "user", "content": user_input})
        self.memory.add({"role": "assistant", "content": response})
        self.logger.info("request=%r response=%r", user_input, response[:200])

    def is_dangerous_request(self, text: str) -> bool:
        return Config.is_dangerous(text)

    def is_valid_response(self, text):
        if not text or len(text) < 20:
            return False
        words = text.split()
        if len(words) < 3:
            return False
        russian_chars = sum(1 for c in text if "а" <= c.lower() <= "я")
        if russian_chars < 3:
            return False
        return True

    def save_dialog(self, user_input, response):
        if self.is_valid_response(response) and Config.can_write(Config.DATA_DIR / "learning_records.json"):
            self.learning.add(user_input, response, "assistant", verified=False)

    def learn_from_correction(self, user_input, correct_answer):
        if not user_input.strip() or not correct_answer.strip():
            return False
        if not Config.can_write(Config.DIALOGS_FILE):
            return
        Config.DIALOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(Config.DIALOGS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{user_input}\n")
            f.write(f"{correct_answer}\n\n")
        if user_input not in self.learned_commands:
            self.learned_commands.append(user_input)
            self.save_learned_commands()
        self.learning.add(user_input, correct_answer, "human_correction", verified=True)
        self.dialogs.reload()
        self.learned_answers = self._load_learned_answers()
        self.learned_answers[self._normalize_question(user_input)] = correct_answer
        return True

    def run(self, user_input: str, confirmed: bool = False) -> str:
        if self.model is None:
            return "Модель не создана"

        if self.security.locked:
            return "Работа заблокирована после подозрительной активности. Разблокируй вручную."

        if user_input.startswith("правильно"):
            parts = user_input.split("правильно", 1)
            if len(parts) > 1:
                correct_answer = parts[1].strip()
                if self.last_question:
                    if not self.learn_from_correction(self.last_question, correct_answer):
                        return "Исправление не сохранено: проверь ответ и разрешение записи"
                    question = self.last_question
                    self.last_question = None
                    return f"Запомнил: на '{question}' нужно отвечать '{correct_answer}'"
            return "Не понял, что исправлять"

        self.last_question = user_input

        action_result = self.execute_action(user_input)
        if action_result:
            self._remember_exchange(user_input, action_result)
            return action_result

        learned_answer = self._get_learned_answer(user_input)
        if learned_answer:
            self._remember_exchange(user_input, learned_answer)
            return learned_answer

        semantic = self._semantic_answer(user_input)
        if semantic:
            self._remember_exchange(user_input, semantic)
            return semantic

        context = self.memory.get_formatted_context(limit=Config.MEMORY_CONTEXT_LIMIT)
        knowledge_context = self.knowledge.search(user_input, limit=1)
        if knowledge_context and len(knowledge_context) > 400:
            knowledge_context = knowledge_context[:400]

        if self.rag.loaded and Config.RAG_ENABLED:
            prompt = self.rag.build_prompt(
                user_input,
                memory=context,
                knowledge=knowledge_context,
                top_k=Config.RAG_TOP_K,
            )
        else:
            prompt = (
                f"{self.planner.create_system_prompt(self.executor.get_tools_description())}\n\n"
                f"История:\n{context}\n\n"
                f"Знания:\n{knowledge_context}\n\n"
                f"Запрос пользователя: {user_input}\n"
            )

        if self.inference is not None:
            try:
                for _ in range(self.planner.max_steps):
                    response = self.inference.generate(
                        prompt, max_new_tokens=Config.MAX_NEW_TOKENS,
                        temperature=Config.TEMPERATURE, top_k=Config.TOP_K_GEN,
                    )
                    action = self.planner.parse_action(response or "")
                    if action is not None:
                        if action.get("done") is True:
                            result = action.get("result", "")
                            if not isinstance(result, str) or not result.strip():
                                return "Некорректный итог плана"
                            self._remember_exchange(user_input, result)
                            return result
                        tool, args = action.get("tool"), action.get("args", {})
                        if not isinstance(tool, str) or not isinstance(args, dict):
                            return "Некорректная команда инструмента"
                        # Model-generated writes need explicit human authorization.
                        if tool in {"write_file", "append_file", "delete_file", "copy_file",
                                    "move_file", "create_folder", "execute_command", "execute_python"}:
                            return "Автоматические команды изменения файлов и выполнения кода отключены"
                        if self.security.locked:
                            return "Работа заблокирована"
                        result = self.executor.execute(tool, args)
                        prompt += f"\nДействие: {response}\nРезультат инструмента: {result}\n"
                        continue
                    if response and len(response) > 10 and not response.startswith("Я не понял"):
                        self._remember_exchange(user_input, response)
                        return response
                    break
                else:
                    return "Достигнут лимит шагов плана"
            except Exception as e:
                self.logger.error("Own model error: %s", e)

        if self.rag.loaded and Config.RAG_ENABLED:
            fragments = self.rag.retrieve(user_input, top_k=1)
            if fragments:
                frag = fragments[0]
                text = clean_answer(frag.get("text", ""))
                if text and answer_looks_meaningful(text):
                    self._remember_exchange(user_input, text)
                    return text

        return "Не знаю. Напиши 'правильно ответ' чтобы научить меня."

    def chat(self):
        print("=" * 60)
        print("КОМАНДЫ:")
        print("  /help        - показать команды")
        print("  /learn       - обучение на диалогах")
        print("  /teach       - включить запись живого обучения")
        print("  /stop-teach  - выключить запись живого обучения")
        print("  /train-now   - дообучить модель на записанных диалогах")
        print("  /train-corpus- дообучить модель на корпусе")
        print("  /index       - построить индекс по корпусу")
        print("  /rag         - статистика RAG")
        print("  /security    - отчёт безопасности")
        print("  /lock        - заблокировать операции")
        print("  /unlock      - разблокировать")
        print("  /exit        - выход")
        print("  правильно    - исправить ответ")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("Вы: ").strip()
                if not user_input:
                    continue
                if user_input == "/exit":
                    print("До свидания!")
                    break
                if user_input == "/help":
                    self.chat_help()
                    continue
                if user_input == "/learn":
                    self.train_from_dialogs(10)
                    continue
                if user_input == "/teach":
                    self.teaching_mode = True
                    print("Режим обучения включён.\n")
                    continue
                if user_input == "/stop-teach":
                    self.teaching_mode = False
                    print("Режим обучения выключен.\n")
                    continue
                if user_input == "/train-now":
                    self.train_from_dialogs(10)
                    continue
                if user_input == "/train-corpus":
                    print("Обучение на корпусе...")
                    self.train_from_corpus(epochs=1)
                    continue
                if user_input == "/index":
                    print("Индексация корпуса...")
                    self.corpus_index.build(Config.DATA_DIR / "corpus.jsonl")
                    self.rag.reload()
                    print(self.rag.stats())
                    continue
                if user_input == "/rag":
                    print(self.rag.stats())
                    continue
                if user_input == "/security":
                    print(Config.security_report())
                    print(self.security.report())
                    continue
                if user_input == "/lock":
                    self.security.lock("manual")
                    print("Операции заблокированы.\n")
                    continue
                if user_input == "/unlock":
                    self.security.unlock()
                    print("Операции разблокированы.\n")
                    continue
                print("Думаю...")
                response = self.run(user_input, confirmed=False)
                if self.teaching_mode:
                    self.save_dialog(user_input, response)
                print(f"\nАссистент: {response}\n")
            except KeyboardInterrupt:
                print("\nДо свидания!")
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    def chat_help(self):
        print("Команды: /help, /learn, /teach, /stop-teach, /train-now, /train-corpus, /index, /rag, /security, /lock, /unlock, /exit")

    def live(self, max_steps: Optional[int] = None) -> None:
        from life.life_loop import LifeLoop
        loop = LifeLoop(self)
        loop.run(max_steps=max_steps)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--train", nargs="*")
    parser.add_argument("--train-corpus", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--host", default=Config.API_HOST)
    parser.add_argument("--port", type=int, default=Config.API_PORT)
    parser.add_argument("--import-knowledge", nargs="*")
    parser.add_argument("--life", action="store_true")
    parser.add_argument("--life-steps", type=int, default=None)
    parser.add_argument("--security-report", action="store_true")
    parser.add_argument("--build-corpus", nargs="+")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--corpus-max-docs", type=int, default=0)
    parser.add_argument("--corpus-max-chars", type=int, default=0)
    parser.add_argument("--rag-stats", action="store_true")
    args = parser.parse_args()

    if args.security_report:
        print(Config.security_report())
        print(get_security().report())
        return

    if args.build_corpus:
        output = Config.DATA_DIR / "corpus.jsonl"
        build_corpus_jsonl(
            [Path(p) for p in args.build_corpus],
            output,
            max_docs=args.corpus_max_docs,
            max_chars=args.corpus_max_chars,
        )
        return

    if args.build_index:
        idx = CorpusIndex()
        idx.build(Config.DATA_DIR / "corpus.jsonl")
        print(idx.stats())
        return

    if args.rag_stats:
        idx = CorpusIndex()
        loaded = idx.load()
        if not loaded:
            print("Индекс не найден")
            return
        print(idx.stats())
        return

    if args.test:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=Config.BASE_DIR,
        )
        return result.returncode

    codex = MyCodex()

    if args.train_corpus:
        if not codex.load():
            return
        codex.train_from_corpus(epochs=args.epochs)
        return

    if args.train:
        if Config.MODEL_FILE.exists() and not codex.load():
            return
        texts = []
        for file_path in args.train:
            if not Config.can_read(file_path):
                print(f"Чтение запрещено: {file_path}")
                continue
            try:
                content = Path(file_path).read_text(encoding="utf-8")
                texts.extend(b.strip() for b in content.split("\n\n") if b.strip())
                print(f"Загружен: {file_path}")
            except Exception as e:
                print(f"Не удалось прочитать {file_path}: {e}")
        if texts:
            codex.train_from_dialogs(args.epochs, texts=texts)
            return

    if not codex.load():
        print("Ошибка загрузки модели")
        return

    if args.import_knowledge:
        for file_path in args.import_knowledge:
            if not Config.can_read(file_path):
                print(f"Чтение запрещено: {file_path}")
                continue
            print(codex.knowledge.import_file(file_path))
        return

    if args.life:
        codex.live(max_steps=args.life_steps)
        return

    if args.api:
        from api import run_api
        run_api(codex, args.host, args.port)
        return

    if args.query:
        result = codex.run(args.query)
        print(f"\nАссистент: {result}\n")
    else:
        codex.chat()


if __name__ == "__main__":
    main()