import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.semantics import tokenize, keywords
from config import Config


class CorpusIndex:
    def __init__(self, index_path: Optional[Path] = None,
                 docs_path: Optional[Path] = None):
        self.index_path = index_path or (Config.CACHE_DIR / "corpus_index.pkl")
        self.docs_path = docs_path or (Config.CACHE_DIR / "corpus_docs.jsonl")
        self.inverted: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_lengths: List[int] = []
        self.doc_ids: List[int] = []
        self.n_docs = 0
        self.avg_len = 1.0

    def build(self, corpus_path: Path, max_docs: int = 200_000):
        from tools.corpus_loader import CorpusLoader

        print(f"Построение индекса: {corpus_path}")
        loader = CorpusLoader(corpus_path)

        self.inverted = {}
        self.doc_lengths = []
        self.doc_ids = []

        self.docs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.docs_path, "w", encoding="utf-8") as docs_out:
            for doc_id, doc in enumerate(loader):
                if doc_id >= max_docs:
                    break
                text = doc.get("text", "")
                tokens = tokenize(text)
                if not tokens:
                    continue
                tf = Counter(tokens)
                for token, count in tf.items():
                    self.inverted.setdefault(token, []).append((doc_id, count))
                self.doc_lengths.append(len(tokens))
                self.doc_ids.append(doc_id)
                docs_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                if doc_id % 10000 == 0 and doc_id > 0:
                    print(f"  обработано: {doc_id}")

        self.n_docs = len(self.doc_lengths)
        self.avg_len = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 1.0
        )
        self.save()
        print(f"Индекс построен: {self.n_docs} документов, "
              f"{len(self.inverted)} токенов")

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "inverted": self.inverted,
                    "doc_lengths": self.doc_lengths,
                    "doc_ids": self.doc_ids,
                    "n_docs": self.n_docs,
                    "avg_len": self.avg_len,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.inverted = data["inverted"]
            self.doc_lengths = data["doc_lengths"]
            self.doc_ids = data["doc_ids"]
            self.n_docs = data["n_docs"]
            self.avg_len = data["avg_len"]
            return True
        except Exception as error:
            print(f"Не удалось загрузить индекс: {error}")
            return False

    def _idf(self, token: str) -> float:
        postings = self.inverted.get(token)
        if not postings:
            return 0.0
        df = len(postings)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.inverted:
            return []
        q_tokens = keywords(query)
        if not q_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scores: Dict[int, float] = {}

        for token in q_tokens:
            postings = self.inverted.get(token)
            if not postings:
                continue
            idf = self._idf(token)
            for doc_id, tf in postings:
                dl = self.doc_lengths[doc_id]
                denom = tf + k1 * (1 - b + b * dl / self.avg_len)
                if denom <= 0:
                    continue
                score = idf * (tf * (k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        results = []
        wanted = {doc_id for doc_id, _ in ranked}
        found: Dict[int, Dict] = {}

        try:
            with open(self.docs_path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i in wanted:
                        try:
                            found[i] = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                    if len(found) == len(wanted):
                        break
        except OSError:
            return []

        for doc_id, score in ranked:
            doc = found.get(doc_id)
            if not doc:
                continue
            results.append(
                {
                    "score": round(score, 4),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }
            )
        return results

    def stats(self) -> Dict:
        return {
            "n_docs": self.n_docs,
            "n_tokens": len(self.inverted),
            "avg_len": round(self.avg_len, 1),
            "index_size_mb": round(self.index_path.stat().st_size / 1024 / 1024, 1)
            if self.index_path.exists() else 0,
        }