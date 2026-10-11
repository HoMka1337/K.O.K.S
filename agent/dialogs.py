from pathlib import Path
from typing import List, Optional, Tuple

from config import Config
from agent.semantics import (
    SemanticIndex,
    answer_looks_meaningful,
    clean_answer,
    detect_question_type,
)


class DialogStore:
    def __init__(self, files: Optional[List[Path]] = None):
        if files is None:
            files = [
                Config.BASE_DIR / "data" / "dialogs.txt",
                Config.BASE_DIR / "data" / "teacher_dialogs.txt",
            ]
        self.files = [Path(f) for f in files]
        self.pairs: List[Tuple[str, str]] = []
        self._index: Optional[SemanticIndex] = None
        self._load()

    def _load(self) -> None:
        self.pairs = []
        for path in self.files:
            if not path.exists():
                continue
            try:
                blocks = path.read_text(encoding='utf-8').split('\n\n')
            except OSError:
                continue
            for block in blocks:
                lines = [l.strip() for l in block.splitlines() if l.strip()]
                if len(lines) >= 2:
                    q, a = lines[0], lines[1]
                    if len(q) >= 3 and len(a) >= 10:
                        self.pairs.append((q, a))

    def reload(self) -> None:
        self._load()
        self._index = None

    def _ensure_index(self) -> SemanticIndex:
        if self._index is None:
            docs = [(str(i), q) for i, (q, _) in enumerate(self.pairs)]
            self._index = SemanticIndex(docs)
        return self._index

    def find(self, question: str, min_score: float = 0.6) -> Optional[str]:
        if not self.pairs:
            return None
        index = self._ensure_index()
        q_type = detect_question_type(question)

        scored: List[Tuple[float, int]] = []
        for i, (q, _) in enumerate(self.pairs):
            s = index.score(question, i)
            if s <= 0:
                continue
            if q_type != "general" and detect_question_type(q) == q_type:
                s *= 1.3
            scored.append((s, i))

        if not scored:
            return None
        scored.sort(reverse=True)
        best_score, best_idx = scored[0]
        if best_score < min_score:
            return None
        answer = self.pairs[best_idx][1]
        answer = clean_answer(answer)
        if not answer_looks_meaningful(answer):
            return None
        return answer

    def add(self, question: str, answer: str) -> None:
        self.pairs.append((question.strip(), answer.strip()))
        self._index = None