
import random
import re
from typing import List

from config import Config


TOPIC_SEEDS = [
    "Python",
    "C++",
    "JavaScript",
    "алгоритмы",
    "структуры данных",
    "объектно-ориентированное программирование",
    "рекурсия",
    "асимптотическая сложность",
    "динамическое программирование",
    "теория графов",
    "бинарный поиск",
    "хеш-таблицы",
    "компиляторы",
    "операционные системы",
    "Linux",
    "Git",
    "Docker",
    "Kubernetes",
    "SQL",
    "PostgreSQL",
    "SQLite",
    "HTTP",
    "HTTPS",
    "TCP",
    "UDP",
    "DNS",
    "REST API",
    "WebSocket",
    "криптография",
    "кибербезопасность",
    "искусственный интеллект",
    "нейросети",
    "машинное обучение",
    "глубокое обучение",
    "Transformer",
    "attention",
    "токенизация",
    "BPE",
    "эмбеддинги",
    "языковые модели",
    "обучение с подкреплением",
    "градиентный спуск",
    "обратное распространение ошибки",
    "функция потерь",
    "оптимизаторы",
    "регуляризация",
    "переобучение",
    "кросс-валидация",
    "fine-tuning",
    "RAG",
    "векторные базы данных",
    "мультимодальные модели",
    "компьютерное зрение",
    "обработка естественного языка",
    "распознавание речи",
    "синтез речи",
    "генеративные модели",
    "диффузионные модели",
    "математика",
    "теория вероятностей",
    "статистика",
    "математический анализ",
    "линейная алгебра",
    "дискретная математика",
    "логика",
    "физика",
    "квантовая физика",
    "квантовые компьютеры",
    "термодинамика",
    "электромагнетизм",
    "астрономия",
    "астрофизика",
    "космология",
    "чёрные дыры",
    "нейтронные звёзды",
    "экзопланеты",
    "космонавтика",
    "робототехника",
    "электроника",
    "микроконтроллеры",
    "интернет вещей",
    "виртуальная реальность",
    "дополненная реальность",
    "трёхмерная графика",
    "компьютерная анимация",
    "биология",
    "генетика",
    "эволюция",
    "нейробиология",
    "мозг человека",
    "иммунная система",
    "микробиология",
    "экология",
    "климатология",
    "география",
    "геология",
    "химия",
    "биохимия",
    "история",
    "археология",
    "лингвистика",
    "психология",
    "когнитивная психология",
    "философия",
    "этика",
    "экономика",
    "финансы",
    "право",
    "социология",
    "антропология",
    "литература",
    "музыка",
    "архитектура",
    "живопись",
    "фотография",
    "кинематограф",
    "образование",
    "критическое мышление",
    "научный метод",
    "научные исследования"
]


STARTER_GOALS = [
    f"узнать что такое {topic}"
    for topic in TOPIC_SEEDS
]


STOPWORDS = {
    "узнать",
    "узнай",
    "поискать",
    "поищи",
    "найти",
    "найди",
    "научиться",
    "научись",
    "считать",
    "посчитать",
    "рассказать",
    "расскажи",
    "сделать",
    "сделай",
    "понять",
    "пойми",
    "изучить",
    "изучи",
    "прочитать",
    "прочитай",
    "написать",
    "напиши",
    "показать",
    "покажи",
    "отвечать",
    "ответь",
    "интересный",
    "интересная",
    "интересное",
    "интересные",
    "полезный",
    "полезная",
    "полезное",
    "полезные",
    "простой",
    "простая",
    "простое",
    "простые",
    "факт",
    "факты",
    "факта",
    "фактов",
    "совет",
    "советы",
    "пример",
    "примеры",
    "примера",
    "примеров",
    "вопрос",
    "ответ",
    "слово",
    "текст",
    "тема",
    "разобраться",
    "ошибкой",
    "ошибка",
    "ошибки",
    "winerror",
    "urlopen",
    "error",
    "exception",
    "traceback",
    "такое",
    "чтобы",
    "который",
    "которое",
    "себя",
    "себе",
    "можно",
    "нельзя",
    "нужно",
    "надо",
    "хочу",
    "хочешь",
    "хотел",
    "хотела",
    "буду",
    "будешь",
    "знаю",
    "знаешь",
    "думаю",
    "думаешь",
    "вижу",
    "видишь",
    "работает",
    "работают",
    "работал",
    "работала",
    "является",
    "являются",
    "быть",
    "был",
    "была",
    "было",
    "были",
    "есть",
    "нет",
    "или",
    "для",
    "при",
    "про",
    "что",
    "как",
    "когда",
    "где",
    "куда",
    "откуда",
    "почему",
    "зачем",
    "какой",
    "какая",
    "какое",
    "какие",
    "этот",
    "эта",
    "это",
    "эти",
    "весь",
    "вся",
    "всё",
    "все"
}


class Curiosity:
    def __init__(self, journal, goals):
        self.journal = journal
        self.goals = goals
        self._random = random.Random()

    @staticmethod
    def _normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""

        return " ".join(
            text.casefold()
            .replace("ё", "е")
            .split()
        )

    def _known_goals(self):
        return {
            self._normalize(goal.text)
            for goal in self.goals.goals
            if isinstance(goal.text, str)
        }

    def _add_goal(
        self,
        text: str,
        source: str,
        priority: int
    ) -> bool:
        if not isinstance(text, str):
            return False

        text = text.strip()

        if len(text) < 8:
            return False

        if self._looks_like_error(text):
            return False

        normalized = self._normalize(text)

        if normalized in self._known_goals():
            return False

        try:
            self.goals.add(
                text,
                source=source,
                priority=priority
            )

            self.journal.add(
                "goal",
                text,
                source=source
            )

            return True

        except (ValueError, TypeError, OSError):
            return False

    def bootstrap(
        self,
        min_goals: int = 3
    ) -> None:
        if not getattr(
            Config,
            "LIFE_CURIOSITY_ENABLED",
            True
        ):
            return

        try:
            min_goals = int(min_goals)
        except (TypeError, ValueError):
            min_goals = 3

        min_goals = max(
            0,
            min(
                min_goals,
                int(Config.LIFE_MAX_GOALS)
            )
        )

        if len(self.goals.active()) >= min_goals:
            return

        known = self._known_goals()

        available = [
            text
            for text in STARTER_GOALS
            if self._normalize(text) not in known
        ]

        self._random.shuffle(available)

        for text in available:
            if len(self.goals.active()) >= min_goals:
                break

            self._add_goal(
                text,
                source="bootstrap",
                priority=1
            )

        if len(self.goals.active()) < min_goals:
            self._random_topic_goal()

    def _random_topic_goal(self) -> None:
        if not TOPIC_SEEDS:
            return

        topics = list(TOPIC_SEEDS)
        self._random.shuffle(topics)

        known = self._known_goals()

        for topic in topics:
            text = f"узнать что такое {topic}"

            if self._normalize(text) in known:
                continue

            if self._add_goal(
                text,
                source="curiosity",
                priority=1
            ):
                return

    def from_errors(self) -> None:
        if not getattr(
            Config,
            "LIFE_ERRORS_TO_GOALS",
            False
        ):
            return

        errors = self.journal.last_errors(
            limit=3
        )

        for entry in errors:
            content = entry.get(
                "content",
                ""
            )

            if not isinstance(content, str):
                continue

            content = content.strip()

            if not content:
                continue

            self.journal.add(
                "thought",
                f"Обнаружена ошибка: {content[:100]}"
            )

    def from_unknown_words(self, text: str) -> None:
        if not getattr(
            Config,
            "LIFE_CURIOSITY_ENABLED",
            True
        ):
            return

        if not isinstance(text, str):
            return

        if self._looks_like_error(text):
            return

        try:
            limit = int(
                Config.LIFE_CURIOSITY_MAX_NEW_GOALS
            )
        except (ValueError, TypeError):
            limit = 2

        limit = max(
            0,
            min(limit, 5)
        )

        if limit == 0:
            return

        words = re.findall(
            r"[А-Яа-яЁёA-Za-z]{6,}",
            text
        )

        if not words:
            return

        seen = set()
        added = 0

        for word in words:
            normalized = self._normalize(word)

            if normalized in seen:
                continue

            seen.add(normalized)

            if normalized in STOPWORDS:
                continue

            if normalized.isascii() and not normalized.isalpha():
                continue

            if len(normalized) < int(
                getattr(
                    Config,
                    "LIFE_MIN_TOPIC_LEN",
                    4
                )
            ):
                continue

            goal_text = (
                f"узнать что такое {word}"
            )

            if self._add_goal(
                goal_text,
                source="curiosity",
                priority=2
            ):
                added += 1

            if added >= limit:
                break

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        if not isinstance(text, str):
            return False

        normalized = text.casefold()

        markers = (
            "ошибк",
            "traceback",
            "urlopen",
            "winerror",
            "exception",
            "syntaxerror",
            "runtimeerror",
            "поиск не удался",
            "connectionerror",
            "permissionerror"
        )

        return any(
            marker in normalized
            for marker in markers
        )

    def ensure_goal(self) -> None:
        if not getattr(
            Config,
            "LIFE_CURIOSITY_ENABLED",
            True
        ):
            return

        if self.goals.active():
            return

        self.bootstrap(
            min_goals=1
        )

        if not self.goals.active():
            self._random_topic_goal()
