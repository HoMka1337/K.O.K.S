
import argparse
import json
import logging
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from config import Config
from core.model import CodexModel
from core.tokenizer import BPETokenizer
from core.trainer import CodexTrainer, read_checkpoint_shape
from core.learning import LearningStore, prepare_training, question_key
from core.inference import CodexInference

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
from agent.rag import RagEngine
from agent.semantics import answer_looks_meaningful, clean_answer


class MyCodex:
    def __init__(self):
        Config.init()
        Config.save_snapshot()

        Config.LOG_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.basicConfig(
            filename=str(Config.LOG_FILE),
            level=logging.INFO,
            format=Config.LOG_FORMAT,
            datefmt=Config.LOG_DATE_FORMAT,
            encoding="utf-8"
        )

        self.logger = logging.getLogger("koks")
        self.security = get_security()

        self.tokenizer = BPETokenizer()
        self.model = None
        self.trainer = None
        self.inference = None
        self.model_loaded = False

        self.executor = Executor()
        self.planner = Planner()
        self.planner.max_steps = Config.PLANNER_MAX_STEPS

        self.memory = Memory(
            Config.MEMORY_FILE,
            max_size=Config.MEMORY_MAX_SIZE
        )

        self.knowledge = Knowledge(
            Config.KNOWLEDGE_FILE
        )

        self.dialogs = DialogStore()
        self.learning = LearningStore()

        self.corpus_index = CorpusIndex()
        self.rag = RagEngine(self.corpus_index)

        self.last_question = None
        self.learned_commands = []
        self.teaching_mode = False

        self._register_tools()

        self.learned_answers = self._load_learned_answers()
        self.load_learned_commands()

        print("=" * 60)
        print("KoKs")
        print("=" * 60)

    @staticmethod
    def _format_number(number):
        return f"{int(number):,}".replace(",", " ")

    @staticmethod
    def _format_size(size_bytes):
        value = float(size_bytes)

        for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
            if value < 1024:
                return f"{value:.2f} {unit}"

            value /= 1024

        return f"{value:.2f} ПБ"

    def model_info(self) -> str:
        lines = [
            "=" * 60,
            "ИНФОРМАЦИЯ ОБ ИИ KoKs",
            "=" * 60,
            "Тип: локальная авторегрессионная языковая модель",
            "Архитектура: Decoder-only Transformer",
            "Фреймворк: PyTorch",
            "Язык интерфейса: русский",
            "Токенизация: Byte Pair Encoding (BPE)",
            "Механизм внимания: Causal Multi-Head Self-Attention",
            "Позиционное кодирование: Sinusoidal",
            "Активация: GELU",
            "Нормализация: LayerNorm",
            "Сеть прямого распространения: Feed Forward Network",
            "Генерация: авторегрессионная",
            "Обучение: Cross Entropy",
            "Оптимизатор: AdamW",
            "Планировщик скорости обучения: Cosine Annealing",
            "Память: локальная история диалогов",
            "База знаний: локальное хранилище",
            "Поиск: семантический поиск и RAG"
        ]

        model = self.model

        if model is None:
            vocab_size = Config.VOCAB_SIZE
            d_model = Config.D_MODEL
            n_heads = Config.N_HEADS
            n_layers = Config.N_LAYERS
            d_ff = Config.D_FF
            max_len = Config.MAX_LEN
            dropout = Config.DROPOUT

            head_dim = d_model // n_heads

            lines.extend([
                "",
                "ПАРАМЕТРЫ КОНФИГУРАЦИИ",
                f"Размер словаря: {self._format_number(vocab_size)}",
                f"Размер эмбеддингов: {d_model}",
                f"Количество Transformer-блоков: {n_layers}",
                f"Количество голов внимания: {n_heads}",
                f"Размер одной головы внимания: {head_dim}",
                f"Размер скрытого слоя FFN: {d_ff}",
                f"Максимальная длина контекста: {max_len} токенов",
                f"Dropout: {dropout}",
                "Модель в памяти: нет",
                "Фактическое количество параметров: недоступно"
            ])

        else:
            vocab_size = model.token_embedding.num_embeddings
            d_model = model.token_embedding.embedding_dim
            n_layers = len(model.layers)
            max_len = model.max_len

            if model.layers:
                first_layer = model.layers[0]

                n_heads = first_layer.attention.n_heads
                d_ff = first_layer.ff.fc1.out_features
                dropout = first_layer.dropout.p

            else:
                n_heads = Config.N_HEADS
                d_ff = Config.D_FF
                dropout = Config.DROPOUT

            head_dim = d_model // max(1, n_heads)

            total_params = sum(
                parameter.numel()
                for parameter in model.parameters()
            )

            trainable_params = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )

            frozen_params = total_params - trainable_params

            parameter_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            )

            buffer_bytes = sum(
                buffer.numel() * buffer.element_size()
                for buffer in model.buffers()
            )

            embedding_params = sum(
                parameter.numel()
                for parameter in model.token_embedding.parameters()
            )

            transformer_params = sum(
                parameter.numel()
                for parameter in model.layers.parameters()
            )

            output_params = sum(
                parameter.numel()
                for parameter in model.fc_out.parameters()
            )

            first_parameter = next(
                model.parameters(),
                None
            )

            dtype = (
                str(first_parameter.dtype).replace("torch.", "")
                if first_parameter is not None
                else "неизвестно"
            )

            device = (
                str(first_parameter.device)
                if first_parameter is not None
                else str(Config.DEVICE)
            )

            ready = bool(
                getattr(
                    model,
                    "generation_ready",
                    False
                )
            )

            lines.extend([
                "",
                "АРХИТЕКТУРА МОДЕЛИ",
                f"Размер словаря: {self._format_number(vocab_size)}",
                f"Размер эмбеддингов (d_model): {d_model}",
                f"Количество Transformer-блоков: {n_layers}",
                f"Количество голов внимания: {n_heads}",
                f"Размер одной головы внимания: {head_dim}",
                f"Размер скрытого слоя FFN: {d_ff}",
                f"Максимальный контекст: {max_len} токенов",
                f"Dropout: {dropout}",
                "",
                "КОЛИЧЕСТВО ПАРАМЕТРОВ",
                f"Всего параметров: {self._format_number(total_params)}",
                f"Обучаемых параметров: {self._format_number(trainable_params)}",
                f"Замороженных параметров: {self._format_number(frozen_params)}",
                f"Параметры эмбеддингов: {self._format_number(embedding_params)}",
                f"Параметры Transformer-блоков: {self._format_number(transformer_params)}",
                f"Параметры выходного слоя: {self._format_number(output_params)}",
                "",
                "ПАМЯТЬ МОДЕЛИ",
                f"Размер весов в памяти: {self._format_size(parameter_bytes)}",
                f"Размер буферов: {self._format_size(buffer_bytes)}",
                f"Тип весов: {dtype}",
                f"Устройство модели: {device}",
                "",
                "СОСТОЯНИЕ",
                f"Чекпоинт загружен: {'да' if self.model_loaded else 'нет'}",
                f"Генерация доступна: {'да' if ready else 'нет'}",
                f"Формат промпта: {getattr(model, 'prompt_style', 'agent')}"
            ])

        tokenizer_size = len(self.tokenizer)

        lines.extend([
            "",
            "ТОКЕНИЗАТОР",
            "Алгоритм: BPE",
            f"Фактический словарь токенизатора: {self._format_number(tokenizer_size)}",
            f"Максимальный словарь по конфигурации: {self._format_number(Config.VOCAB_SIZE)}",
            f"Количество BPE-объединений: {len(self.tokenizer.merges)}",
            f"Версия формата: {getattr(self.tokenizer, 'format_version', 'неизвестно')}",
            "",
            "ГЕНЕРАЦИЯ ТЕКСТА",
            f"Temperature: {Config.TEMPERATURE}",
            f"Top-K: {Config.TOP_K_GEN}",
            f"Top-P (настройка): {Config.TOP_P_GEN}",
            f"Repetition Penalty (настройка): {Config.REPETITION_PENALTY}",
            f"Максимум новых токенов: {Config.MAX_NEW_TOKENS}",
            "",
            "ОБУЧЕНИЕ",
            f"Batch Size: {Config.TRAIN_BATCH_SIZE}",
            f"Training Sequence Length: {Config.TRAIN_MAX_LEN}",
            f"Learning Rate: {Config.LEARNING_RATE}",
            f"Количество эпох по умолчанию: {Config.EPOCHS}",
            f"Gradient Clipping: {Config.GRAD_CLIP}",
            f"Weight Decay: {Config.WEIGHT_DECAY}",
            f"Автообучение: {'включено' if Config.LIFE_AUTOLEARN_EVERY > 0 else 'отключено'}",
            "",
            "ОБОРУДОВАНИЕ",
            f"Устройство PyTorch: {Config.DEVICE}",
            f"Версия PyTorch: {torch.__version__}",
            f"CUDA доступна: {'да' if torch.cuda.is_available() else 'нет'}"
        ])

        if torch.cuda.is_available():
            try:
                gpu = torch.cuda.get_device_properties(0)

                lines.extend([
                    f"Видеокарта: {torch.cuda.get_device_name(0)}",
                    f"Видеопамять: {self._format_size(gpu.total_memory)}",
                    f"Вычислительная способность CUDA: {gpu.major}.{gpu.minor}",
                    f"Выделено VRAM: {self._format_size(torch.cuda.memory_allocated())}",
                    f"Зарезервировано VRAM: {self._format_size(torch.cuda.memory_reserved())}"
                ])

            except (RuntimeError, AssertionError):
                pass

        system = Config.SYSTEM_INFO

        total_ram = system.get("total_ram_gb")
        ram_text = (
            f"{total_ram} ГБ"
            if total_ram is not None
            else "неизвестно"
        )

        lines.extend([
            f"Операционная система: {system.get('platform', 'неизвестно')}",
            f"Процессор: {system.get('processor') or 'неизвестно'}",
            f"Логических процессоров: {system.get('cpu_count', 'неизвестно')}",
            f"Оперативная память: {ram_text}",
            "",
            "ДОПОЛНИТЕЛЬНЫЕ СИСТЕМЫ",
            f"RAG: {'включён' if Config.RAG_ENABLED else 'отключён'}",
            f"Индекс RAG: {'загружен' if self.rag.loaded else 'не загружен'}",
            f"Интернет: {'разрешён' if Config.ALLOW_NETWORK else 'отключён'}",
            f"Безопасный режим: {'включён' if Config.SAFE_MODE else 'отключён'}",
            "=" * 60
        ])

        return "\n".join(lines)

    def show_model_info(self):
        print(self.model_info())

    @staticmethod
    def _normalize_question(text: str) -> str:
        return question_key(text)

    def load_learned_commands(self):
        self.learned_commands = []

        path = Config.LEARNED_COMMANDS_FILE

        if not path.is_file():
            return

        try:
            self.learned_commands = [
                line.strip()
                for line in path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        except (OSError, UnicodeError):
            self.learned_commands = []

    def save_learned_commands(self):
        path = Config.LEARNED_COMMANDS_FILE

        if not Config.can_write(path):
            return False

        try:
            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            path.write_text(
                "\n".join(self.learned_commands),
                encoding="utf-8"
            )

            return True

        except OSError:
            return False

    def _load_learned_answers(self):
        answers = {}

        for path in (
            Config.TEACHER_DIALOGS_FILE,
            Config.DIALOGS_FILE
        ):
            if not path.is_file():
                continue

            try:
                content = path.read_text(
                    encoding="utf-8"
                )

            except (OSError, UnicodeError):
                continue

            for block in re.split(
                r"\r?\n\s*\r?\n",
                content
            ):
                question, separator, answer = (
                    block.strip().partition("\n")
                )

                if not separator:
                    continue

                key = self._normalize_question(question)
                answer = answer.strip()

                if key and answer:
                    answers[key] = answer

        for row in self.learning.verified():
            question = row.get("question", "")
            answer = row.get("answer", "")

            key = self._normalize_question(question)

            if key and isinstance(answer, str) and answer.strip():
                answers[key] = answer.strip()

        return answers

    def _get_learned_answer(
        self,
        question: str
    ) -> Optional[str]:
        key = self._normalize_question(question)

        if not key:
            return None

        return self.learned_answers.get(key)

    def _semantic_answer(
        self,
        question: str
    ) -> Optional[str]:
        answer = self.dialogs.find(
            question,
            min_score=0.6
        )

        if answer:
            return answer

        answer = self.knowledge.search(
            question,
            limit=1
        )

        return answer or None

    def _register_tools(self):
        tools = [
            (
                "read_file",
                FileTools.read_file,
                "Чтение файла"
            ),
            (
                "write_file",
                FileTools.write_file,
                "Запись файла"
            ),
            (
                "append_file",
                FileTools.append_file,
                "Добавление текста в файл"
            ),
            (
                "list_files",
                FileTools.list_files,
                "Список файлов"
            ),
            (
                "delete_file",
                FileTools.delete_file,
                "Удаление файла"
            ),
            (
                "copy_file",
                FileTools.copy_file,
                "Копирование файла"
            ),
            (
                "move_file",
                FileTools.move_file,
                "Перемещение файла"
            ),
            (
                "create_folder",
                FileTools.create_folder,
                "Создание папки"
            ),
            (
                "execute_command",
                SystemTools.execute_command,
                "Системные команды"
            ),
            (
                "execute_python",
                CodeTools.execute_python,
                "Выполнение Python"
            ),
            (
                "analyze_python",
                CodeTools.analyze_python,
                "Анализ Python"
            ),
            (
                "web_search",
                WebTools.search,
                "Поиск в интернете"
            ),
            (
                "search_knowledge",
                self.knowledge.search,
                "Поиск по знаниям"
            )
        ]

        for name, function, description in tools:
            self.executor.register_tool(
                name,
                function,
                description
            )

    def create_model(
        self,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_len: Optional[int] = None
    ):
        if vocab_size is None:
            vocab_size = max(
                len(self.tokenizer),
                Config.VOCAB_SIZE
            )

        d_model = int(d_model or Config.D_MODEL)
        n_heads = int(n_heads or Config.N_HEADS)
        n_layers = int(n_layers or Config.N_LAYERS)
        d_ff = int(d_ff or Config.D_FF)
        max_len = int(max_len or Config.MAX_LEN)

        if d_model % n_heads != 0:
            n_heads = next(
                (
                    candidate
                    for candidate in (8, 4, 2, 1)
                    if d_model % candidate == 0
                ),
                1
            )

        self.model = CodexModel(
            vocab_size=int(vocab_size),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=Config.DROPOUT
        )

        self.model.generation_ready = False
        self.model_loaded = False

        self.trainer = CodexTrainer(
            self.model,
            self.tokenizer,
            Config.DEVICE
        )

        self.inference = CodexInference(
            self.model,
            self.tokenizer,
            Config.DEVICE
        )

        return self.model

    def _initialize_training_model(self, texts):
        if self.trainer is not None:
            return

        if Config.MODEL_FILE.is_file():
            if not self.load(show_info=False):
                raise RuntimeError(
                    "Не удалось загрузить модель"
                )
            return

        if Config.TOKENIZER_FILE.is_file():
            self.tokenizer.load(
                str(Config.TOKENIZER_FILE)
            )

        else:
            self.tokenizer.train(
                texts,
                vocab_size=Config.VOCAB_SIZE
            )

            self.tokenizer.save(
                str(Config.TOKENIZER_FILE)
            )

        self.create_model(
            vocab_size=len(self.tokenizer)
        )

    def train_from_dialogs(
        self,
        epochs=10,
        texts=None
    ):
        if texts is None:
            teacher_path = Config.TEACHER_DIALOGS_FILE

            if teacher_path.is_file():
                content = teacher_path.read_text(
                    encoding="utf-8"
                )

                verified_keys = {
                    question_key(row.get("question", ""))
                    for row in self.learning.verified()
                }

                for block in re.split(
                    r"\r?\n\s*\r?\n",
                    content
                ):
                    question, separator, answer = (
                        block.strip().partition("\n")
                    )

                    key = question_key(question)

                    if (
                        separator
                        and key
                        and answer.strip()
                        and key not in verified_keys
                    ):
                        self.learning.add(
                            question,
                            answer,
                            "teacher_dialogs.txt",
                            verified=True
                        )

                        verified_keys.add(key)

            rows = self.learning.batch(
                Config.LIFE_MAX_TRAIN_TEXTS
            )

            texts = [
                f"{row['question']}\n{row['answer']}"
                for row in rows
                if isinstance(row, dict)
                and row.get("question")
                and row.get("answer")
            ]

        if not texts:
            print("Нет подходящих примеров для обучения")
            return False

        try:
            training, validation = prepare_training(
                texts,
                Config.LIFE_MAX_TRAIN_TEXTS
            )

            self._initialize_training_model(
                training + validation
            )

            accepted = self.trainer.train(
                texts=training,
                epochs=max(1, int(epochs)),
                batch_size=min(
                    Config.TRAIN_BATCH_SIZE,
                    max(1, len(training))
                ),
                lr=Config.LEARNING_RATE,
                max_len=min(
                    Config.TRAIN_MAX_LEN,
                    self.model.max_len
                ),
                save_path=str(Config.MODEL_FILE),
                validation_texts=validation
            )

            if accepted:
                self.model.generation_ready = True
                self.model_loaded = True

                self.trainer.save(
                    str(Config.MODEL_FILE)
                )

                self.learning.mark_trained(training)

                self.show_model_info()

            return bool(accepted)

        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError
        ) as error:
            self.logger.exception(
                "Training failed: %s",
                error
            )

            print(f"Ошибка обучения: {error}")
            return False

    def train_from_corpus(
        self,
        epochs=1,
        max_texts=None
    ):
        path = Config.DATA_DIR / "corpus.jsonl"

        if not path.is_file():
            print(f"Корпус не найден: {path}")
            return False

        limit = max(
            1,
            int(max_texts or Config.LIFE_MAX_TRAIN_TEXTS)
        )

        sampler = random.Random(Config.RANDOM_SEED)
        texts = []
        seen = 0

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace"
        ) as stream:
            for line in stream:
                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict):
                    continue

                text = record.get("text", "")

                if not isinstance(text, str):
                    continue

                text = text.strip()

                if len(text) < 60:
                    continue

                seen += 1

                if len(texts) < limit:
                    texts.append(text)

                else:
                    index = sampler.randrange(seen)

                    if index < limit:
                        texts[index] = text

        if not texts:
            print("Корпус не содержит подходящих текстов")
            return False

        return self.train_from_dialogs(
            epochs=epochs,
            texts=texts
        )

    def load(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        show_info: bool = True
    ):
        model_path = Path(
            model_path or Config.MODEL_FILE
        )

        tokenizer_path = Path(
            tokenizer_path or Config.TOKENIZER_FILE
        )

        if not model_path.is_file():
            if tokenizer_path.is_file():
                try:
                    self.tokenizer.load(
                        str(tokenizer_path)
                    )

                except (
                    OSError,
                    ValueError,
                    TypeError,
                    json.JSONDecodeError
                ):
                    return False

            self.create_model(
                vocab_size=max(
                    len(self.tokenizer),
                    Config.VOCAB_SIZE
                )
            )

            if show_info:
                self.show_model_info()

            return True

        if not tokenizer_path.is_file():
            print("Токенизатор для сохранённой модели не найден")
            return False

        try:
            self.tokenizer.load(
                str(tokenizer_path)
            )

            shape = read_checkpoint_shape(
                str(model_path)
            )

            if not isinstance(shape, dict):
                return False

            required = (
                "vocab_size",
                "d_model",
                "n_heads",
                "n_layers",
                "d_ff",
                "max_len"
            )

            if any(
                shape.get(key) is None
                for key in required
            ):
                return False

            if shape["vocab_size"] != len(self.tokenizer):
                print(
                    "Размеры словаря модели и токенизатора не совпадают"
                )
                return False

            self.create_model(
                vocab_size=shape["vocab_size"],
                d_model=shape["d_model"],
                n_heads=shape["n_heads"],
                n_layers=shape["n_layers"],
                d_ff=shape["d_ff"],
                max_len=shape["max_len"]
            )

            self.trainer.load(
                str(model_path)
            )

            self.model_loaded = True

            if show_info:
                self.show_model_info()

            return True

        except Exception as error:
            self.logger.exception(
                "Model loading failed: %s",
                error
            )

            self.model = None
            self.trainer = None
            self.inference = None
            self.model_loaded = False

            print(f"Ошибка загрузки модели: {error}")
            return False

    def _confirm_dangerous(
        self,
        action: str,
        confirmed: bool = False
    ) -> bool:
        if Config.READ_ONLY_MODE:
            return False

        if not Config.CONFIRM_DANGEROUS:
            return True

        if confirmed and not Config.REQUIRE_CONFIRMATION_CODE:
            return True

        if not sys.stdin.isatty():
            return False

        print(f"Подтверждение действия: {action}")

        try:
            if Config.REQUIRE_CONFIRMATION_CODE:
                code = self.security.generate_confirmation_code(
                    action
                )

                if not code:
                    return False

                print(f"Код подтверждения: {code}")

                entered = input("Код: ").strip()

                return self.security.verify_confirmation_code(
                    entered
                )

            response = input(
                "Подтвердить? (да/нет): "
            ).strip().casefold()

            return response in {
                "да",
                "д",
                "yes",
                "y"
            }

        except (EOFError, KeyboardInterrupt):
            return False

    def execute_action(
        self,
        text: str,
        confirmed: bool = False
    ) -> Optional[str]:
        if not isinstance(text, str):
            return None

        original = text.strip()
        lowered = original.casefold()

        if lowered in {
            "/info",
            "/model-info",
            "/model",
            "информация об ии",
            "информация о модели",
            "характеристики ии",
            "характеристики модели",
            "сколько параметров",
            "сколько параметров у тебя",
            "какая у тебя архитектура",
            "что ты используешь"
        }:
            return self.model_info()

        greeting = lowered.strip("!?.,:; ")

        if greeting in {
            "привет",
            "здравствуй",
            "здравствуйте",
            "ку",
            "хай",
            "салют"
        }:
            return random.choice([
                "Привет!",
                "Здравствуйте!",
                "Ку! Чем займёмся?",
                "Привет! Что будем делать?"
            ])

        for prefix in (
            "поищи в интернете",
            "найди в интернете"
        ):
            if lowered.startswith(prefix):
                query = original[len(prefix):].strip()

                if not query:
                    return "Укажи поисковый запрос"

                return WebTools.search(query)

        if lowered in {
            "который час",
            "сколько времени",
            "время"
        }:
            return datetime.now().strftime(
                "Сейчас %H:%M:%S"
            )

        if lowered in {
            "какая сегодня дата",
            "какое сегодня число",
            "дата"
        }:
            return datetime.now().strftime(
                "Сегодня %d.%m.%Y"
            )

        if lowered in {
            "информация о системе",
            "системная информация",
            "характеристики компьютера"
        }:
            return SystemTools.get_system_info()

        if lowered.startswith("запомни это"):
            match = re.search(
                r"https?://\S+",
                original
            )

            if not match:
                return "Укажи ссылку"

            url = match.group(0).rstrip(".,);]")

            if not Config.can_write(Config.KNOWLEDGE_FILE):
                return "Сохранение знаний запрещено"

            self.knowledge.add_url(url)

            return f"Запомнил ссылку: {url}"

        if lowered.startswith("запомни файл"):
            filename = original[len("запомни файл"):].strip()

            if not filename:
                return "Укажи имя файла"

            if not Config.can_read(filename):
                return "Чтение запрещено"

            if not Config.can_write(Config.KNOWLEDGE_FILE):
                return "Сохранение знаний запрещено"

            return self.knowledge.import_file(filename)

        match = re.fullmatch(
            r"(?:создай файл|создай текстовый документ)\s+"
            r"(\S+)(?:\s+(?:с текстом|текст:)\s*(.*))?",
            original,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            filename = match.group(1)
            content = match.group(2) or ""

            target = Config.normalize_path(filename)

            if not Config.can_write(target):
                return f"Запись запрещена: {filename}"

            if not self._confirm_dangerous(
                f"Создать файл {filename}",
                confirmed
            ):
                return "Действие отменено"

            return FileTools.write_file(
                str(target),
                content
            )

        if lowered.startswith(
            ("покажи файлы", "список файлов")
        ):
            prefix = (
                "покажи файлы"
                if lowered.startswith("покажи файлы")
                else "список файлов"
            )

            folder = original[len(prefix):].strip() or "."

            return FileTools.list_files(folder)

        if lowered.startswith("прочитай файл"):
            filename = original[len("прочитай файл"):].strip()

            if not filename:
                return "Укажи имя файла"

            return FileTools.read_file(filename)

        if lowered.startswith("найди файл"):
            filename = original[len("найди файл"):].strip()

            if not filename:
                return "Укажи имя файла"

            if (
                Path(filename).name != filename
                or any(
                    character in filename
                    for character in "*?[]"
                )
            ):
                return "Некорректное имя файла"

            matches = []

            try:
                for path in Config.BASE_DIR.rglob(filename):
                    if Config.can_read(path):
                        matches.append(str(path))

                    if len(matches) >= 20:
                        break

            except OSError as error:
                return f"Ошибка поиска: {error}"

            return "\n".join(matches) or "Файл не найден"

        match = re.fullmatch(
            r"(?:создай|сделай)\s+папку\s+(.+)",
            original,
            flags=re.IGNORECASE
        )

        if match:
            folder = match.group(1).strip()
            target = Config.normalize_path(folder)

            if not Config.can_write(target):
                return f"Создание папки запрещено: {folder}"

            if not self._confirm_dangerous(
                f"Создать папку {folder}",
                confirmed
            ):
                return "Действие отменено"

            return FileTools.create_folder(
                str(target)
            )

        if lowered.startswith(
            ("удали файл", "удали папку")
        ):
            return "Удаление отключено политикой безопасности"

        if lowered.startswith(
            ("выполни команду", "выполни код")
        ):
            return "Выполнение команд отключено политикой безопасности"

        if lowered.startswith(
            ("открой файл", "открой папку")
        ):
            return "Открытие программ ОС отключено"

        if any(
            phrase in lowered
            for phrase in (
                "посчитай",
                "реши",
                "сколько будет"
            )
        ):
            match = re.search(
                r"(-?\d+(?:[.,]\d+)?)\s*"
                r"([+\-*/])\s*"
                r"(-?\d+(?:[.,]\d+)?)",
                original
            )

            if match:
                first = float(
                    match.group(1).replace(",", ".")
                )

                second = float(
                    match.group(3).replace(",", ".")
                )

                operation = match.group(2)

                if operation == "+":
                    result = first + second
                elif operation == "-":
                    result = first - second
                elif operation == "*":
                    result = first * second
                else:
                    if second == 0:
                        return "На ноль делить нельзя"

                    result = first / second

                return str(result)

        return None

    def _remember_exchange(
        self,
        user_input: str,
        response: str
    ) -> None:
        self.memory.add({
            "role": "user",
            "content": user_input
        })

        self.memory.add({
            "role": "assistant",
            "content": response
        })

        self.logger.info(
            "request=%r response=%r",
            user_input,
            response[:200]
        )

    def is_dangerous_request(self, text: str) -> bool:
        return Config.is_dangerous(text)

    def is_valid_response(self, text) -> bool:
        return (
            isinstance(text, str)
            and len(text.strip()) >= 20
            and len(text.split()) >= 3
        )

    def save_dialog(self, user_input, response):
        if not self.is_valid_response(response):
            return False

        if not Config.can_write(self.learning.path):
            return False

        return self.learning.add(
            user_input,
            response,
            "assistant",
            verified=False
        )

    def learn_from_correction(
        self,
        user_input,
        correct_answer
    ):
        if not isinstance(user_input, str):
            return False

        if not isinstance(correct_answer, str):
            return False

        question = user_input.strip()
        answer = correct_answer.strip()

        if not question or not answer:
            return False

        if not Config.can_write(self.learning.path):
            return False

        try:
            self.learning.add(
                question,
                answer,
                "human_correction",
                verified=True
            )

            self.learned_answers = self._load_learned_answers()

            self.learned_answers[
                self._normalize_question(question)
            ] = answer

            if question not in self.learned_commands:
                self.learned_commands.append(question)
                self.save_learned_commands()

            return True

        except (OSError, ValueError, TypeError):
            return False

    def run(
        self,
        user_input: str,
        confirmed: bool = False
    ) -> str:
        if not isinstance(user_input, str):
            return "Запрос должен быть строкой"

        user_input = user_input.strip()

        if not user_input:
            return "Пустой запрос"

        if self.security.locked:
            return "Работа заблокирована"

        correction = re.fullmatch(
            r"правильно(?:\s+(.+))?",
            user_input,
            flags=re.IGNORECASE | re.DOTALL
        )

        if correction:
            answer = (correction.group(1) or "").strip()

            if not answer or not self.last_question:
                return "Не понял, что исправлять"

            question = self.last_question

            if not self.learn_from_correction(
                question,
                answer
            ):
                return "Исправление не сохранено"

            self.last_question = None

            return (
                f"Запомнил: на '{question}' "
                f"нужно отвечать '{answer}'"
            )

        self.last_question = user_input

        action_result = self.execute_action(
            user_input,
            confirmed=confirmed
        )

        if action_result is not None:
            self._remember_exchange(
                user_input,
                action_result
            )
            return action_result

        learned = self._get_learned_answer(user_input)

        if learned is not None:
            self._remember_exchange(
                user_input,
                learned
            )
            return learned

        semantic = self._semantic_answer(user_input)

        if semantic:
            self._remember_exchange(
                user_input,
                semantic
            )
            return semantic

        if (
            self.model is None
            or self.inference is None
            or not getattr(
                self.model,
                "generation_ready",
                False
            )
        ):
            return (
                "Пока не знаю ответа. "
                "Напиши 'правильно ответ', чтобы научить меня."
            )

        memory_context = self.memory.get_formatted_context(
            limit=Config.MEMORY_CONTEXT_LIMIT
        )

        knowledge_context = self.knowledge.search(
            user_input,
            limit=1
        )

        if not isinstance(knowledge_context, str):
            knowledge_context = str(knowledge_context or "")

        knowledge_context = knowledge_context[:400]

        if self.rag.loaded and Config.RAG_ENABLED:
            prompt = self.rag.build_prompt(
                user_input,
                memory=memory_context,
                knowledge=knowledge_context,
                top_k=Config.RAG_TOP_K
            )

        elif getattr(
            self.model,
            "prompt_style",
            "agent"
        ) == "plain_dialog":
            prompt = user_input

        else:
            prompt = (
                self.planner.create_system_prompt(
                    self.executor.get_tools_description()
                )
                + f"\nИстория:\n{memory_context}\n"
                + f"Знания:\n{knowledge_context}\n"
                + f"Запрос пользователя: {user_input}\n"
            )

        try:
            for _ in range(
                max(1, self.planner.max_steps)
            ):
                response = self.inference.generate(
                    prompt,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    temperature=Config.TEMPERATURE,
                    top_k=Config.TOP_K_GEN
                )

                action = self.planner.parse_action(
                    response or ""
                )

                if action is not None:
                    if action.get("done") is True:
                        result = action.get("result", "")

                        if not isinstance(result, str):
                            return "Некорректный результат"

                        result = result.strip()

                        if not result:
                            return "Пустой результат"

                        self._remember_exchange(
                            user_input,
                            result
                        )

                        return result

                    tool = action.get("tool")
                    args = action.get("args", {})

                    blocked_tools = {
                        "write_file",
                        "append_file",
                        "delete_file",
                        "copy_file",
                        "move_file",
                        "create_folder",
                        "execute_command",
                        "execute_python"
                    }

                    if tool in blocked_tools:
                        return (
                            "Автоматические изменяющие команды отключены"
                        )

                    if tool not in self.executor.tools:
                        return "Неизвестный инструмент"

                    if not isinstance(args, dict):
                        return "Некорректные аргументы инструмента"

                    result = self.executor.execute(
                        tool,
                        args
                    )

                    prompt += (
                        f"\nДействие: {response}\n"
                        f"Результат инструмента: {result}\n"
                    )

                    continue

                if isinstance(response, str) and response.strip():
                    response = response.strip()

                    self._remember_exchange(
                        user_input,
                        response
                    )

                    return response

                break

        except Exception as error:
            self.logger.exception(
                "Generation failed: %s",
                error
            )

        if self.rag.loaded and Config.RAG_ENABLED:
            fragments = self.rag.retrieve(
                user_input,
                top_k=1
            )

            if fragments:
                text = clean_answer(
                    fragments[0].get("text", "")
                )

                if text and answer_looks_meaningful(text):
                    self._remember_exchange(
                        user_input,
                        text
                    )

                    return text

        return (
            "Не знаю. Напиши 'правильно ответ', "
            "чтобы научить меня."
        )

    def chat_help(self):
        print(
            "/help, /info, /model-info, /learn, /teach, "
            "/stop-teach, /train-now, /train-corpus, "
            "/index, /rag, /security, /lock, /unlock, /exit"
        )

    def chat(self):
        self.chat_help()

        while True:
            try:
                user_input = input("Вы: ").strip()

                if not user_input:
                    continue

                if user_input == "/exit":
                    break

                if user_input == "/help":
                    self.chat_help()
                    continue

                if user_input in {
                    "/info",
                    "/model-info",
                    "/model"
                }:
                    self.show_model_info()
                    continue

                if user_input == "/teach":
                    self.teaching_mode = True
                    print("Режим обучения включён")
                    continue

                if user_input == "/stop-teach":
                    self.teaching_mode = False
                    print("Режим обучения выключен")
                    continue

                if user_input in {
                    "/learn",
                    "/train-now"
                }:
                    self.train_from_dialogs(10)
                    continue

                if user_input == "/train-corpus":
                    self.train_from_corpus(
                        epochs=1
                    )
                    continue

                if user_input == "/index":
                    self.corpus_index.build(
                        Config.DATA_DIR / "corpus.jsonl"
                    )

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
                    print("Заблокировано")
                    continue

                if user_input == "/unlock":
                    self.security.unlock()
                    print("Разблокировано")
                    continue

                response = self.run(user_input)

                if self.teaching_mode:
                    self.save_dialog(
                        user_input,
                        response
                    )

                print(f"Ассистент: {response}")

            except (KeyboardInterrupt, EOFError):
                break

            except Exception as error:
                self.logger.exception(
                    "Chat failed: %s",
                    error
                )

                print(f"Ошибка: {error}")

    def live(
        self,
        max_steps: Optional[int] = None
    ):
        from life.life_loop import LifeLoop

        loop = LifeLoop(self)
        loop.run(max_steps=max_steps)


def main():
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
    parser.add_argument("--life-steps", type=int)
    parser.add_argument("--security-report", action="store_true")
    parser.add_argument("--model-info", action="store_true")
    parser.add_argument("--build-corpus", nargs="+")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--corpus-max-docs", type=int, default=0)
    parser.add_argument("--corpus-max-chars", type=int, default=0)
    parser.add_argument("--rag-stats", action="store_true")

    args = parser.parse_args()

    if args.security_report:
        print(Config.security_report())
        print(get_security().report())
        return 0

    if args.test:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v"
            ],
            cwd=str(Config.BASE_DIR),
            check=False
        )

        return result.returncode

    if args.build_corpus:
        count = build_corpus_jsonl(
            [
                Path(path)
                for path in args.build_corpus
            ],
            Config.DATA_DIR / "corpus.jsonl",
            max_docs=args.corpus_max_docs,
            max_chars=args.corpus_max_chars
        )

        print(f"Документов: {count}")
        return 0

    if args.build_index:
        index = CorpusIndex()

        index.build(
            Config.DATA_DIR / "corpus.jsonl"
        )

        print(index.stats())
        return 0

    if args.rag_stats:
        index = CorpusIndex()

        if index.load():
            print(index.stats())
        else:
            print("Индекс не найден")

        return 0

    codex = MyCodex()

    if args.model_info:
        if not codex.load():
            return 1

        return 0

    if args.train:
        if Config.MODEL_FILE.exists() and not codex.load(
            show_info=False
        ):
            return 1

        texts = []

        for filename in args.train:
            if not Config.can_read(filename):
                print(f"Чтение запрещено: {filename}")
                continue

            try:
                content = Path(filename).read_text(
                    encoding="utf-8"
                )

                texts.extend(
                    block.strip()
                    for block in re.split(
                        r"\r?\n\s*\r?\n",
                        content
                    )
                    if block.strip()
                )

            except (OSError, UnicodeError) as error:
                print(f"Ошибка чтения: {error}")

        if not texts:
            return 1

        return (
            0
            if codex.train_from_dialogs(
                args.epochs,
                texts=texts
            )
            else 1
        )

    if not codex.load(
        show_info=not args.train_corpus
    ):
        print("Ошибка загрузки модели")
        return 1

    if args.train_corpus:
        return (
            0
            if codex.train_from_corpus(
                epochs=args.epochs
            )
            else 1
        )

    if args.import_knowledge:
        for filename in args.import_knowledge:
            if Config.can_read(filename):
                print(
                    codex.knowledge.import_file(filename)
                )
            else:
                print(f"Чтение запрещено: {filename}")

        return 0

    if args.life:
        codex.live(
            max_steps=args.life_steps
        )
        return 0

    if args.api:
        from api import run_api

        run_api(
            codex,
            args.host,
            args.port
        )

        return 0

    if args.query:
        print(codex.run(args.query))
        return 0

    codex.chat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
