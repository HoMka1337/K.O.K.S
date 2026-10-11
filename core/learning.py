
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
from pathlib import Path

from config import Config


_LOCK = threading.RLock()


def question_key(text):
    if not isinstance(text, str):
        return ""

    question = text.split("\n", 1)[0]
    question = question.casefold().replace("ё", "е")
    question = re.sub(r"[^\w\s]", " ", question, flags=re.UNICODE)

    return " ".join(question.split())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        suffix=".tmp"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2
            )
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        Path(temporary).unlink(missing_ok=True)


class LearningStore:
    def __init__(self, path=None):
        self.path = Path(
            path if path is not None
            else Config.DATA_DIR / "learning_records.json"
        )

    def records(self):
        with _LOCK:
            if not self.path.exists():
                return []

            try:
                data = json.loads(
                    self.path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                return []

            if not isinstance(data, list):
                return []

            return [
                row for row in data
                if isinstance(row, dict)
            ]

    def add(self, question, answer, source, verified=False):
        if not isinstance(question, str):
            return False

        if not isinstance(answer, str):
            return False

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            return False

        text = question + "\n" + answer
        identity = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        now = time.time()

        with _LOCK:
            records = self.records()

            for row in records:
                if row.get("id") != identity:
                    continue

                if verified and (
                    not row.get("verified", False)
                    or row.get("source") != source
                ):
                    row.update(
                        verified=True,
                        source=source,
                        updated=now
                    )
                    atomic_json(self.path, records)
                    return True

                return False

            records.append({
                "id": identity,
                "question": question,
                "answer": answer,
                "source": source,
                "verified": bool(verified),
                "updated": now,
                "trained": 0
            })

            atomic_json(self.path, records)

        return True

    def verified(self):
        latest = {}

        rows = sorted(
            self.records(),
            key=lambda row: float(row.get("updated", 0) or 0)
        )

        for row in rows:
            if not row.get("verified", False):
                continue

            question = row.get("question", "")
            answer = row.get("answer", "")

            if not isinstance(question, str):
                continue

            if not isinstance(answer, str):
                continue

            key = question_key(question)

            if key and answer.strip():
                latest[key] = row

        return list(latest.values())

    def get_answer(self, question):
        key = question_key(question)

        if not key:
            return None

        for row in self.verified():
            if question_key(row.get("question", "")) == key:
                answer = row.get("answer", "")

                if isinstance(answer, str) and answer.strip():
                    return answer.strip()

        return None

    def batch(self, limit, require_new=False):
        if limit <= 0:
            return []

        rows = self.verified()
        validation_path = Config.DATA_DIR / "validation.json"

        excluded = set()

        if validation_path.exists():
            try:
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )

                if isinstance(validation, list):
                    excluded = {
                        question_key(text)
                        for text in validation
                        if isinstance(text, str)
                    }

            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        rows = [
            row for row in rows
            if question_key(row.get("question", "")) not in excluded
        ]

        fresh = [
            row for row in rows
            if row.get("trained", 0) == 0
        ]

        fresh.sort(
            key=lambda row: row.get("updated", 0),
            reverse=True
        )

        old = [
            row for row in rows
            if row.get("trained", 0) > 0
        ]

        if require_new and len(fresh) < Config.LIFE_MIN_GOOD_PAIRS:
            return []

        random.Random(42).shuffle(old)

        if not old:
            return fresh[:limit]

        fresh_count = min(
            len(fresh),
            max(1, limit // 5),
            max(1, len(old) // 4)
        )

        return (
            fresh[:fresh_count]
            + old[:max(0, limit - fresh_count)]
        )

    def mark_trained(self, texts):
        identities = {
            hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
            for text in texts
            if isinstance(text, str)
        }

        with _LOCK:
            records = self.records()
            changed = False

            for row in records:
                if row.get("id") in identities:
                    row["trained"] = int(
                        row.get("trained", 0)
                    ) + 1
                    changed = True

            if changed:
                atomic_json(self.path, records)


def prepare_training(texts, limit):
    unique = {}

    for text in texts:
        if not isinstance(text, str):
            continue

        text = text.strip()

        if text:
            key = question_key(text)

            if key:
                unique[key] = text

    texts = list(unique.values())

    path = Config.DATA_DIR / "validation.json"

    with _LOCK:
        if path.exists():
            validation = json.loads(
                path.read_text(encoding="utf-8")
            )
        else:
            if len(texts) < 10:
                raise ValueError(
                    "Need at least 10 distinct reviewed examples "
                    "to create validation.json"
                )

            shuffled = list(texts)
            random.Random(42).shuffle(shuffled)

            validation = shuffled[
                :max(2, len(shuffled) // 10)
            ]

            atomic_json(path, validation)

    if (
        not isinstance(validation, list)
        or not validation
        or not all(
            isinstance(text, str) and text.strip()
            for text in validation
        )
    ):
        raise ValueError(
            "validation.json must contain a nonempty "
            "JSON array of texts"
        )

    held_out = {
        question_key(text)
        for text in validation
    }

    training = [
        text for text in texts
        if question_key(text) not in held_out
    ]

    if len(training) > limit:
        recent_count = max(1, limit // 5)

        recent = training[-recent_count:]
        older = training[:-recent_count]

        random.Random(42).shuffle(older)

        training = (
            older[:limit - recent_count]
            + recent
        )

    if not training:
        raise ValueError(
            "No examples outside the fixed validation set"
        )

    return training, validation
