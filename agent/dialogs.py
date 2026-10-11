
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from config import Config
from agent.semantics import (
    SemanticIndex,
    answer_looks_meaningful,
    clean_answer,
    detect_question_type,
    tokenize,
)


class DialogStore:
    def __init__(self, files: Optional[List[Path]] = None):
        if files is None:
            files = [
                Config.BASE_DIR / "data" / "dialogs.txt",
                Config.BASE_DIR / "data" / "teacher_dialogs.txt",
            ]

        self.files = [Path(path) for path in files]
        self.pairs: List[Tuple[str, str]] = []
        self._index: Optional[SemanticIndex] = None
        self._lock = threading.RLock()

        self._load()

    def _normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.casefold().replace("ё", "е")
        text = re.sub(r"[^\w\s]", " ", text)

        return " ".join(text.split())

    def _load(self) -> None:
        pairs = []

        for path in self.files:
            if not path.is_file():
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue

            blocks = re.split(
                r"\r?\n\s*\r?\n",
                content
            )

            for block in blocks:
                lines = [
                    line.strip()
                    for line in block.splitlines()
                    if line.strip()
                ]

                if len(lines) < 2:
                    continue

                question = lines[0]
                answer = "\n".join(lines[1:])

                if not question or not answer:
                    continue

                pairs.append((question, answer))

        with self._lock:
            self.pairs = pairs
            self._index = None

    def reload(self) -> None:
        self._load()

    def _ensure_index(self) -> SemanticIndex:
        with self._lock:
            if self._index is None:
                documents = [
                    (str(index), question)
                    for index, (question, _) in enumerate(self.pairs)
                ]

                self._index = SemanticIndex(documents)

            return self._index

    def find(
        self,
        question: str,
        min_score: float = 0.6
    ) -> Optional[str]:
        if not isinstance(question, str):
            return None

        question = question.strip()

        if not question:
            return None

        normalized = self._normalize(question)

        with self._lock:
            if not self.pairs:
                return None

            for saved_question, answer in reversed(self.pairs):
                if self._normalize(saved_question) == normalized:
                    return answer.strip() or None

            index = self._ensure_index()
            question_type = detect_question_type(question)
            question_tokens = set(tokenize(question))

            if not question_tokens:
                return None

            best_score = 0.0
            best_index = -1

            for position, (saved_question, answer) in enumerate(
                self.pairs
            ):
                saved_tokens = set(tokenize(saved_question))

                if not saved_tokens:
                    continue

                overlap = question_tokens & saved_tokens

                if not overlap:
                    continue

                coverage = len(overlap) / max(
                    1,
                    len(question_tokens)
                )

                if coverage < 0.5:
                    continue

                score = index.score(
                    question,
                    position
                )

                if score <= 0:
                    continue

                saved_type = detect_question_type(
                    saved_question
                )

                if (
                    question_type != "general"
                    and saved_type != "general"
                    and question_type != saved_type
                ):
                    continue

                if (
                    question_type != "general"
                    and saved_type == question_type
                ):
                    score *= 1.3

                if score > best_score:
                    best_score = score
                    best_index = position

            if best_index < 0:
                return None

            if best_score < min_score:
                return None

            answer = self.pairs[best_index][1]
            answer = clean_answer(answer)

            if not answer_looks_meaningful(answer):
                return None

            return answer

    def add(self, question: str, answer: str) -> None:
        if not isinstance(question, str):
            return

        if not isinstance(answer, str):
            return

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            return

        normalized = self._normalize(question)

        with self._lock:
            self.pairs = [
                (saved_question, saved_answer)
                for saved_question, saved_answer in self.pairs
                if self._normalize(saved_question) != normalized
            ]

            self.pairs.append((question, answer))
            self._index = None
