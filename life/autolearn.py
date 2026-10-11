import re
from pathlib import Path
from typing import List, Optional

from config import Config
from core.learning import LearningStore


BAD_QUESTION_MARKERS = (
    "ошибка", "error", "traceback", "urlopen", "winerror",
    "поиск не удался", "не удался", "exception",
)

BAD_QUESTIONS_EXACT = {
    "узнать", "поискать", "интересный", "полезный", "научиться",
    "рассказывать", "отвечать", "считать", "внимание", "машинное",
    "узнать что такое узнать", "узнать что такое поискать",
    "узнать что такое интересный", "узнать что такое полезный",
    "узнать что такое научиться", "узнать что такое рассказывать",
    "узнать что такое отвечать", "узнать что такое считать",
}

BAD_ANSWER_STARTS = (
    "перейти к содержанию", "главное меню", "узнать инн",
    "welcome to python.org notice", "яндекс яндекс",
    "перейти к содержимому",
)

BAD_ANSWER_CONTAINS = (
    "главное меню переместить", "служебные страницы",
    "пожертвовать создать учётную запись", "полный индекс категории",
    "требуется согласие на обработку персональных данных",
)

STOP_WORDS = {
    "что", "такое", "как", "это", "для", "или", "и", "в", "на",
    "с", "по", "из", "у", "о", "же", "бы", "ли", "не",
}


class AutoLearner:
    def __init__(self, codex, journal):
        self.codex = codex
        self.journal = journal
        self.learning = LearningStore()

    def maybe_train(self, step: int) -> None:
        if Config.LIFE_AUTOLEARN_EVERY <= 0:
            return
        if step == 0 or step % Config.LIFE_AUTOLEARN_EVERY != 0:
            return
        rows = self.learning.batch(Config.LIFE_MAX_TRAIN_TEXTS, require_new=True)
        pairs = [row["question"] + "\n" + row["answer"] for row in rows]
        if len(pairs) < Config.LIFE_MIN_GOOD_PAIRS:
            self.journal.add("thought", f"мало хороших пар для обучения: {len(pairs)}")
            return
        self.journal.add("thought", f"автообучение на {len(pairs)} примерах")
        try:
            accepted = self.codex.train_from_dialogs(
                epochs=Config.LIFE_AUTOLEARN_EPOCHS,
                texts=pairs,
            )
            self.journal.add("thought", "кандидат принят" if accepted else "обучение отложено или кандидат отклонён", pairs=len(pairs))
        except Exception as error:
            self.journal.add("error", f"автообучение: {error}")

    def _is_good_pair(self, pair: str) -> bool:
        lines = [l.strip() for l in pair.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        question, answer = lines[0], lines[1]
        return self._is_good_question(question) and self._is_good_answer(answer)

    @staticmethod
    def _is_good_question(question: str) -> bool:
        q = question.strip()
        low = q.lower()
        if len(q) < 4 or len(q) > 120:
            return False
        if low in BAD_QUESTIONS_EXACT:
            return False
        if any(m in low for m in BAD_QUESTION_MARKERS):
            return False
        words = re.findall(r"[А-Яа-яA-Za-z]{3,}", low)
        if len(words) == 1 and words[0] in {
            "узнать", "поискать", "научиться", "рассказать",
            "отвечать", "считать", "показать",
        }:
            return False
        return True

    @staticmethod
    def _is_good_answer(answer: str) -> bool:
        a = answer.strip()
        if len(a) < 60 or len(a) > 1200:
            return False
        low = a.lower()
        if any(low.startswith(s) for s in BAD_ANSWER_STARTS):
            return False
        if any(s in low for s in BAD_ANSWER_CONTAINS):
            return False
        if a.count("http") >= 2 or "duckduckgo.com/l/" in a:
            return False
        ru = sum(1 for c in a if 'а' <= c.lower() <= 'я')
        en = sum(1 for c in a if 'a' <= c.lower() <= 'z')
        if ru + en < 40:
            return False
        return True

    def is_meaningful_question(self, question: str) -> bool:
        return self._is_good_question(question)

    @staticmethod
    def extract_answer_from_page(question: str, page_text: str) -> Optional[str]:
        if not page_text:
            return None
        text = re.sub(r"\s+", " ", page_text).strip()
        if len(text) < 100:
            return None

        sentences = re.split(r"(?<=[.!?])\s+", text)
        q_words = {
            w.lower() for w in re.findall(r"[А-Яа-яA-Za-z]{4,}", question)
            if w.lower() not in STOP_WORDS
        }
        if not q_words:
            return None

        scored = []
        for i, s in enumerate(sentences):
            s_low = s.lower()
            score = sum(1 for w in q_words if w in s_low)
            if score > 0:
                scored.append((score, i, s))

        if not scored:
            return None

        scored.sort(key=lambda x: (-x[0], x[1]))
        _, best_idx, best_sentence = scored[0]
        context = [best_sentence]
        if best_idx + 1 < len(sentences):
            context.append(sentences[best_idx + 1])
        answer = " ".join(context).strip()
        if len(answer) < 60:
            return None
        return answer[:800]

    def learn_from_page(self, question: str, page_text: str, source: str = "web") -> bool:
        if not self.is_meaningful_question(question):
            return False
        answer = self.extract_answer_from_page(question, page_text)
        if not answer or not self._is_good_answer(answer):
            return False
        if not Config.LIFE_ALLOW_WRITES or not Config.can_write(Config.DATA_DIR / "learning_records.json"):
            return False
        saved = self.learning.add(question, answer, source, verified=False)
        if saved:
            self.journal.add("candidate", "фрагмент сохранён для проверки", question=question,
                             answer=answer, source=source, verified=False)
        return saved

    def learn_from_search_results(self, question: str, search_text: str) -> bool:
        return False
