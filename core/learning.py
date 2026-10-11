"""Reviewed examples, persistent holdout, and replay selection."""
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from pathlib import Path
from config import Config

_LOCK = threading.RLock()


def question_key(text):
    return " ".join(text.split("\n", 1)[0].casefold().split())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class LearningStore:
    def __init__(self, path=None):
        self.path = Path(path or Config.DATA_DIR / "learning_records.json")

    def records(self):
        with _LOCK:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def add(self, question, answer, source, verified=False):
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            return False
        text = question + "\n" + answer
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with _LOCK:
            records = self.records()
            for row in records:
                if row["id"] == identity:
                    if verified and not row["verified"]:
                        row.update(verified=True, source=source, updated=time.time())
                        atomic_json(self.path, records)
                        return True
                    return False
            records.append(dict(id=identity, question=question, answer=answer,
                                source=source, verified=verified, updated=time.time(), trained=0))
            atomic_json(self.path, records)
        return True

    def verified(self):
        latest = {}
        for row in sorted(self.records(), key=lambda r: r["updated"]):
            if row["verified"]:
                latest[question_key(row["question"])] = row
        return list(latest.values())

    def batch(self, limit, require_new=False):
        rows = self.verified()
        validation_path = Config.DATA_DIR / "validation.json"
        excluded = {question_key(t) for t in json.loads(validation_path.read_text(encoding="utf-8"))} if validation_path.exists() else set()
        rows = [r for r in rows if question_key(r["question"]) not in excluded]
        fresh = [row for _, row in sorted(
            ((i, row) for i, row in enumerate(rows) if row["trained"] == 0),
            key=lambda item: (item[1]["updated"], item[0]), reverse=True)]
        old = [r for r in rows if r["trained"] > 0]
        if require_new and len(fresh) < Config.LIFE_MIN_GOOD_PAIRS:
            return []
        random.Random(42).shuffle(old)
        if not old:
            return fresh[:limit]
        fresh_count = min(len(fresh), max(1, limit // 5), max(1, len(old) // 4))
        return fresh[:fresh_count] + old[:max(0, limit - fresh_count)]

    def mark_trained(self, texts):
        identities = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts}
        with _LOCK:
            records = self.records()
            for row in records:
                if row["id"] in identities:
                    row["trained"] += 1
            atomic_json(self.path, records)


def prepare_training(texts, limit):
    # Latest correction wins; validation questions are excluded, including changed answers.
    unique = {}
    for text in texts:
        if text.strip():
            unique[question_key(text)] = text.strip()
    texts = list(unique.values())
    path = Config.DATA_DIR / "validation.json"
    with _LOCK:
        if path.exists():
            validation = json.loads(path.read_text(encoding="utf-8"))
        else:
            if len(texts) < 10:
                raise ValueError("Need at least 10 distinct reviewed examples to create validation.json")
            shuffled = list(texts)
            random.Random(42).shuffle(shuffled)
            validation = shuffled[:max(2, len(shuffled) // 10)]
            atomic_json(path, validation)
    if not isinstance(validation, list) or not validation or not all(isinstance(t, str) and t.strip() for t in validation):
        raise ValueError("validation.json must contain a nonempty JSON array of texts")
    held_out = {question_key(t) for t in validation}
    training = [t for t in texts if question_key(t) not in held_out]
    if len(training) > limit:
        recent_count = max(1, limit // 5)
        recent = training[-recent_count:]
        older = training[:-recent_count]
        random.Random(42).shuffle(older)
        training = older[:limit - recent_count] + recent
    if not training:
        raise ValueError("No examples outside the fixed validation set")
    return training, validation
