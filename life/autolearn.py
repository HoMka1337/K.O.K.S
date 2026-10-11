
import re
from typing import Optional

from config import Config
from core.learning import LearningStore, question_key


BAD_QUESTION_MARKERS = (
    "ошибка",
    "traceback",
    "urlopen",
    "winerror",
    "поиск не удался",
    "exception",
    "syntaxerror",
    "runtimeerror"
)

BAD_QUESTIONS_EXACT = {
    "узнать",
    "поискать",
    "интересный",
    "полезный",
    "научиться",
    "рассказывать",
    "отвечать",
    "считать",
    "внимание",
    "машинное",
    "узнать что такое узнать",
    "узнать что такое поискать",
    "узнать что такое интересный",
    "узнать что такое полезный",
    "узнать что такое научиться",
    "узнать что такое рассказывать",
    "узнать что такое отвечать",
    "узнать что такое считать"
}

BAD_ANSWER_STARTS = (
    "перейти к содержанию",
    "главное меню",
    "узнать инн",
    "welcome to python.org notice",
    "яндекс яндекс",
    "перейти к содержимому"
)

BAD_ANSWER_CONTAINS = (
    "главное меню переместить",
    "служебные страницы",
    "пожертвовать создать учётную запись",
    "полный индекс категории",
    "требуется согласие на обработку персональных данных"
)

STOP_WORDS = {
    "что", "такое", "как", "это", "для",
    "или", "и", "в", "на", "с", "по",
    "из", "у", "о", "же", "бы", "ли",
    "не", "про", "об", "при", "когда",
    "почему", "зачем", "где"
}


class AutoLearner:
    def __init__(self, codex, journal):
        self.codex = codex
        self.journal = journal
        self.learning = LearningStore()

    def maybe_train(self, step: int) -> None:
        interval = int(
            getattr(Config, "LIFE_AUTOLEARN_EVERY", 0)
        )

        if interval <= 0:
            return

        if not isinstance(step, int) or step <= 0:
            return

        if step % interval != 0:
            return

        if self.codex is None:
            return

        limit = max(
            1,
            int(Config.LIFE_MAX_TRAIN_TEXTS)
        )

        minimum = max(
            1,
            int(Config.LIFE_MIN_GOOD_PAIRS)
        )

        rows = self.learning.batch(
            limit,
            require_new=True
        )

        pairs = []
        seen = set()

        for row in rows:
            if not isinstance(row, dict):
                continue

            if row.get("verified") is not True:
                continue

            question = row.get("question")
            answer = row.get("answer")

            if not isinstance(question, str):
                continue

            if not isinstance(answer, str):
                continue

            question = question.strip()
            answer = answer.strip()

            pair = f"{question}\n{answer}"

            if not self._is_good_pair(pair):
                continue

            key = question_key(question)

            if key in seen:
                continue

            seen.add(key)
            pairs.append(pair)

        if len(pairs) < minimum:
            self.journal.add(
                "thought",
                f"Недостаточно проверенных примеров: {len(pairs)}"
            )
            return

        self.journal.add(
            "thought",
            f"Начато обучение на {len(pairs)} проверенных примерах"
        )

        try:
            accepted = self.codex.train_from_dialogs(
                epochs=max(
                    1,
                    int(Config.LIFE_AUTOLEARN_EPOCHS)
                ),
                texts=pairs
            )

            self.journal.add(
                "learn" if accepted else "thought",
                (
                    "Обучение принято"
                    if accepted
                    else "Обучение отложено или отклонено"
                ),
                pairs=len(pairs),
                accepted=bool(accepted),
                verified=False
            )

        except Exception as error:
            self.journal.add(
                "error",
                f"Ошибка обучения: {error}"
            )

    def _is_good_pair(self, pair: str) -> bool:
        if not isinstance(pair, str):
            return False

        question, separator, answer = pair.partition("\n")

        if not separator:
            return False

        return (
            self._is_good_question(question)
            and self._is_good_answer(answer)
        )

    @staticmethod
    def _is_good_question(question: str) -> bool:
        if not isinstance(question, str):
            return False

        question = question.strip()

        if not 4 <= len(question) <= 200:
            return False

        normalized = " ".join(
            question.casefold().replace("ё", "е").split()
        )

        if normalized in BAD_QUESTIONS_EXACT:
            return False

        if any(
            marker in normalized
            for marker in BAD_QUESTION_MARKERS
        ):
            return False

        words = re.findall(
            r"[а-яa-z]{3,}",
            normalized
        )

        if not words:
            return False

        if len(words) == 1 and words[0] in {
            "узнать",
            "поискать",
            "научиться",
            "рассказать",
            "отвечать",
            "считать",
            "показать"
        }:
            return False

        return True

    @staticmethod
    def _is_good_answer(answer: str) -> bool:
        if not isinstance(answer, str):
            return False

        answer = answer.strip()

        minimum = max(
            1,
            int(
                getattr(
                    Config,
                    "LIFE_MIN_ANSWER_FOR_PAIR",
                    60
                )
            )
        )

        maximum = max(
            minimum,
            int(
                getattr(
                    Config,
                    "LIFE_MAX_ANSWER_FOR_PAIR",
                    1500
                )
            )
        )

        if not minimum <= len(answer) <= maximum:
            return False

        normalized = answer.casefold()

        if any(
            normalized.startswith(marker)
            for marker in BAD_ANSWER_STARTS
        ):
            return False

        if any(
            marker in normalized
            for marker in BAD_ANSWER_CONTAINS
        ):
            return False

        if answer.count("http://") + answer.count("https://") >= 2:
            return False

        if "duckduckgo.com/l/" in normalized:
            return False

        letters = sum(
            character.isalpha()
            for character in answer
        )

        if letters < 40:
            return False

        return True

    def is_meaningful_question(self, question: str) -> bool:
        return self._is_good_question(question)

    @staticmethod
    def extract_answer_from_page(
        question: str,
        page_text: str
    ) -> Optional[str]:
        if not isinstance(question, str):
            return None

        if not isinstance(page_text, str):
            return None

        question = question.strip()

        if not AutoLearner._is_good_question(question):
            return None

        text = re.sub(
            r"\s+",
            " ",
            page_text
        ).strip()

        if len(text) < 100:
            return None

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        words = {
            word
            for word in re.findall(
                r"[а-яa-z]{4,}",
                question.casefold().replace("ё", "е")
            )
            if word not in STOP_WORDS
        }

        if not words:
            return None

        scored = []

        for index, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if not sentence:
                continue

            normalized = sentence.casefold().replace(
                "ё",
                "е"
            )

            sentence_words = set(
                re.findall(
                    r"[а-яa-z]{4,}",
                    normalized
                )
            )

            overlap = len(
                words & sentence_words
            )

            if overlap == 0:
                continue

            score = overlap / len(words)

            scored.append(
                (score, overlap, index)
            )

        if not scored:
            return None

        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2]
            )
        )

        best_score, overlap, best_index = scored[0]

        if best_score < 0.25:
            return None

        context = [
            sentences[best_index].strip()
        ]

        if best_index + 1 < len(sentences):
            context.append(
                sentences[best_index + 1].strip()
            )

        answer = " ".join(context).strip()

        if len(answer) < 60:
            return None

        return answer[:800].strip()

    def learn_from_page(
        self,
        question: str,
        page_text: str,
        source: str = "web"
    ) -> bool:
        if not self.is_meaningful_question(question):
            return False

        answer = self.extract_answer_from_page(
            question,
            page_text
        )

        if not answer:
            return False

        if not self._is_good_answer(answer):
            return False

        if not getattr(
            Config,
            "LIFE_ALLOW_WRITES",
            False
        ):
            return False

        if getattr(Config, "READ_ONLY_MODE", False):
            return False

        if not Config.can_write(
            self.learning.path
        ):
            return False

        source = str(source or "web").strip()

        try:
            saved = self.learning.add(
                question,
                answer,
                source,
                verified=False
            )

        except (OSError, ValueError, TypeError):
            return False

        if saved:
            self.journal.add(
                "candidate",
                "Фрагмент сохранён для проверки",
                question=question,
                answer=answer,
                source=source,
                verified=False
            )

        return bool(saved)

    def learn_from_search_results(
        self,
        question: str,
        search_text: str
    ) -> bool:
        return False
