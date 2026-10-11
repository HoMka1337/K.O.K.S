from typing import List, Optional

from agent.corpus_index import CorpusIndex
from agent.semantics import clean_answer, answer_looks_meaningful
from config import Config


class RagEngine:
    def __init__(self, corpus_index: Optional[CorpusIndex] = None):
        self.index = corpus_index or CorpusIndex()
        self.loaded = self.index.load()

    def reload(self):
        self.loaded = self.index.load()

    def retrieve(self, question: str, top_k: int = 3) -> List[dict]:
        if not self.loaded:
            return []
        return self.index.search(question, limit=top_k)

    def build_prompt(self, question: str, memory: str = "",
                     knowledge: str = "", top_k: int = 3) -> str:
        fragments = self.retrieve(question, top_k=top_k)
        context_parts = []
        for i, frag in enumerate(fragments, 1):
            text = clean_answer(frag.get("text", ""))
            if not text or not answer_looks_meaningful(text):
                continue
            title = frag.get("title", "")
            context_parts.append(f"[{i}] {title}: {text}")

        context = "\n".join(context_parts) if context_parts else "Нет контекста"

        prompt = (
            "Ты — ассистент KoKs. Отвечай на вопрос пользователя, опираясь "
            "на фрагменты контекста ниже. Если в контексте нет ответа — "
            "скажи, что не знаешь.\n\n"
            f"Контекст:\n{context}\n\n"
            f"История:\n{memory or 'Нет истории'}\n\n"
            f"Знания:\n{knowledge or 'Нет знаний'}\n\n"
            f"Вопрос: {question}\n"
            f"Ответ:"
        )
        return prompt

    def stats(self) -> dict:
        if not self.loaded:
            return {"loaded": False}
        stats = self.index.stats()
        stats["loaded"] = True
        return stats