
import gzip
import json
import math
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.semantics import keywords, tokenize
from config import Config


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            "Global objects are not allowed"
        )

    def persistent_load(self, pid):
        raise pickle.UnpicklingError(
            "Persistent objects are not allowed"
        )


class CorpusIndex:
    def __init__(
        self,
        index_path: Optional[Path] = None,
        docs_path: Optional[Path] = None
    ):
        self.index_path = Path(
            index_path
            if index_path is not None
            else Config.CACHE_DIR / "corpus_index.pkl"
        )

        self.docs_path = Path(
            docs_path
            if docs_path is not None
            else Config.CACHE_DIR / "corpus_docs.jsonl"
        )

        self.inverted: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_lengths: List[int] = []
        self.doc_ids: List[int] = []
        self.n_docs = 0
        self.avg_len = 1.0

    def _reset(self):
        self.inverted = {}
        self.doc_lengths = []
        self.doc_ids = []
        self.n_docs = 0
        self.avg_len = 1.0

    def _payload(self):
        return {
            "format_version": 2,
            "inverted": self.inverted,
            "doc_lengths": self.doc_lengths,
            "doc_ids": self.doc_ids,
            "n_docs": self.n_docs,
            "avg_len": self.avg_len
        }

    def _write_index(self, path, payload):
        with gzip.open(
            path,
            "wt",
            encoding="utf-8",
            compresslevel=5
        ) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                separators=(",", ":")
            )

    def _temporary_path(self, parent, prefix):
        fd, name = tempfile.mkstemp(
            prefix=prefix,
            suffix=".tmp",
            dir=str(parent)
        )

        os.close(fd)
        return Path(name)

    def build(
        self,
        corpus_path: Path,
        max_docs: int = 200_000
    ):
        from tools.corpus_loader import CorpusLoader

        max_docs = max(0, int(max_docs))

        self.docs_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_docs = self._temporary_path(
            self.docs_path.parent,
            ".corpus_docs_"
        )

        temporary_index = self._temporary_path(
            self.index_path.parent,
            ".corpus_index_"
        )

        inverted = {}
        doc_lengths = []

        try:
            loader = CorpusLoader(Path(corpus_path))

            with temporary_docs.open(
                "w",
                encoding="utf-8"
            ) as docs_out:
                for document in loader:
                    if len(doc_lengths) >= max_docs:
                        break

                    if not isinstance(document, dict):
                        continue

                    text = document.get("text", "")

                    if not isinstance(text, str):
                        continue

                    tokens = tokenize(text)

                    if not tokens:
                        continue

                    document_id = len(doc_lengths)
                    frequencies = Counter(tokens)

                    serialized = json.dumps(
                        document,
                        ensure_ascii=False
                    )

                    docs_out.write(serialized + "\n")

                    doc_lengths.append(len(tokens))

                    for token, frequency in frequencies.items():
                        inverted.setdefault(
                            token,
                            []
                        ).append(
                            (document_id, frequency)
                        )

                docs_out.flush()
                os.fsync(docs_out.fileno())

            n_docs = len(doc_lengths)

            avg_len = (
                sum(doc_lengths) / n_docs
                if n_docs
                else 1.0
            )

            payload = {
                "format_version": 2,
                "inverted": inverted,
                "doc_lengths": doc_lengths,
                "doc_ids": list(range(n_docs)),
                "n_docs": n_docs,
                "avg_len": avg_len
            }

            self._write_index(
                temporary_index,
                payload
            )

            os.replace(
                temporary_docs,
                self.docs_path
            )

            os.replace(
                temporary_index,
                self.index_path
            )

            self.inverted = inverted
            self.doc_lengths = doc_lengths
            self.doc_ids = list(range(n_docs))
            self.n_docs = n_docs
            self.avg_len = avg_len

            return self.stats()

        finally:
            temporary_docs.unlink(
                missing_ok=True
            )

            temporary_index.unlink(
                missing_ok=True
            )

    def save(self):
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary = self._temporary_path(
            self.index_path.parent,
            ".corpus_index_"
        )

        try:
            self._write_index(
                temporary,
                self._payload()
            )

            os.replace(
                temporary,
                self.index_path
            )

        finally:
            temporary.unlink(
                missing_ok=True
            )

    def _read_payload(self):
        with self.index_path.open("rb") as stream:
            header = stream.read(2)

        if header == b"\x1f\x8b":
            with gzip.open(
                self.index_path,
                "rt",
                encoding="utf-8"
            ) as stream:
                return json.load(stream)

        with self.index_path.open("rb") as stream:
            return SafeUnpickler(stream).load()

    def _validate_payload(self, data):
        if not isinstance(data, dict):
            raise ValueError("Invalid index data")

        inverted = data.get("inverted")
        doc_lengths = data.get("doc_lengths")
        doc_ids = data.get("doc_ids")

        if not isinstance(inverted, dict):
            raise ValueError("Invalid inverted index")

        if not isinstance(doc_lengths, list):
            raise ValueError("Invalid document lengths")

        if not isinstance(doc_ids, list):
            raise ValueError("Invalid document IDs")

        lengths = []

        for value in doc_lengths:
            if isinstance(value, bool):
                raise ValueError("Invalid document length")

            length = int(value)

            if length <= 0:
                raise ValueError("Invalid document length")

            lengths.append(length)

        n_docs = len(lengths)

        if len(doc_ids) != n_docs:
            raise ValueError("Document count mismatch")

        original_ids = []

        for value in doc_ids:
            if isinstance(value, bool):
                raise ValueError("Invalid document ID")

            document_id = int(value)

            if document_id < 0:
                raise ValueError("Invalid document ID")

            original_ids.append(document_id)

        if len(set(original_ids)) != n_docs:
            raise ValueError("Duplicate document IDs")

        mapping = {
            document_id: position
            for position, document_id in enumerate(original_ids)
        }

        validated_inverted = {}

        for token, postings in inverted.items():
            if not isinstance(token, str):
                raise ValueError("Invalid index token")

            if not isinstance(postings, list):
                raise ValueError("Invalid postings")

            validated_postings = []
            seen = set()

            for posting in postings:
                if not isinstance(posting, (list, tuple)):
                    raise ValueError("Invalid posting")

                if len(posting) != 2:
                    raise ValueError("Invalid posting")

                document_id, frequency = posting

                if isinstance(document_id, bool):
                    raise ValueError("Invalid posting ID")

                if isinstance(frequency, bool):
                    raise ValueError("Invalid frequency")

                document_id = int(document_id)
                frequency = int(frequency)

                if document_id not in mapping:
                    raise ValueError("Unknown document ID")

                position = mapping[document_id]

                if position in seen:
                    raise ValueError("Duplicate posting")

                if frequency <= 0:
                    raise ValueError("Invalid frequency")

                if frequency > lengths[position]:
                    raise ValueError("Invalid frequency")

                seen.add(position)

                validated_postings.append(
                    (position, frequency)
                )

            validated_inverted[token] = validated_postings

        avg_len = (
            sum(lengths) / n_docs
            if n_docs
            else 1.0
        )

        return {
            "inverted": validated_inverted,
            "doc_lengths": lengths,
            "doc_ids": list(range(n_docs)),
            "n_docs": n_docs,
            "avg_len": avg_len
        }

    def load(self) -> bool:
        if not self.index_path.is_file():
            self._reset()
            return False

        if not self.docs_path.is_file():
            self._reset()
            return False

        try:
            data = self._read_payload()
            validated = self._validate_payload(data)

            self.inverted = validated["inverted"]
            self.doc_lengths = validated["doc_lengths"]
            self.doc_ids = validated["doc_ids"]
            self.n_docs = validated["n_docs"]
            self.avg_len = validated["avg_len"]

            return True

        except (
            OSError,
            EOFError,
            ValueError,
            TypeError,
            KeyError,
            UnicodeError,
            json.JSONDecodeError,
            pickle.UnpicklingError
        ):
            self._reset()
            return False

    def _idf(self, token: str) -> float:
        postings = self.inverted.get(token)

        if not postings or self.n_docs <= 0:
            return 0.0

        df = len(postings)

        return math.log(
            1.0 + (
                self.n_docs - df + 0.5
            ) / (
                df + 0.5
            )
        )

    def search(
        self,
        query: str,
        limit: int = 3
    ) -> List[Dict]:
        if not isinstance(query, str):
            return []

        query = query.strip()

        if not query:
            return []

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return []

        if limit <= 0:
            return []

        if self.n_docs <= 0 or not self.inverted:
            return []

        query_tokens = keywords(query)

        if not query_tokens:
            return []

        k1 = 1.5
        b = 0.75

        scores: Dict[int, float] = {}

        for token in query_tokens:
            postings = self.inverted.get(token)

            if not postings:
                continue

            idf = self._idf(token)

            for document_id, frequency in postings:
                if not 0 <= document_id < self.n_docs:
                    continue

                document_length = self.doc_lengths[
                    document_id
                ]

                denominator = (
                    frequency
                    + k1 * (
                        1.0 - b
                        + b * document_length / self.avg_len
                    )
                )

                if denominator <= 0:
                    continue

                score = (
                    idf
                    * frequency
                    * (k1 + 1.0)
                    / denominator
                )

                scores[document_id] = (
                    scores.get(document_id, 0.0)
                    + score
                )

        if not scores:
            return []

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                item[0]
            )
        )[:limit]

        wanted = {
            document_id
            for document_id, _ in ranked
        }

        found = {}

        try:
            with self.docs_path.open(
                "r",
                encoding="utf-8",
                errors="replace"
            ) as stream:
                for position, line in enumerate(stream):
                    if position not in wanted:
                        continue

                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(document, dict):
                        found[position] = document

                    if len(found) == len(wanted):
                        break

        except OSError:
            return []

        results = []

        for document_id, score in ranked:
            document = found.get(document_id)

            if document is None:
                continue

            title = document.get("title", "")
            content = document.get("text", "")

            if not isinstance(title, str):
                title = str(title)

            if not isinstance(content, str):
                continue

            results.append({
                "score": round(score, 4),
                "title": title,
                "text": content
            })

        return results

    def stats(self) -> Dict:
        try:
            index_size = (
                self.index_path.stat().st_size
                if self.index_path.is_file()
                else 0
            )
        except OSError:
            index_size = 0

        return {
            "n_docs": self.n_docs,
            "n_tokens": len(self.inverted),
            "avg_len": round(self.avg_len, 1),
            "index_size_mb": round(
                index_size / 1024 / 1024,
                1
            )
        }
