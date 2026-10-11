
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config


class Goal:
    def __init__(
        self,
        text: str,
        source: str = "self",
        priority: int = 1,
        created_at: Optional[float] = None,
        done: bool = False,
        result: str = ""
    ):
        if not isinstance(text, str):
            raise TypeError("Goal text must be a string")

        self.text = text.strip()
        self.source = str(source or "self")

        try:
            self.priority = int(priority)
        except (TypeError, ValueError, OverflowError):
            self.priority = 1

        self.priority = max(
            0,
            min(self.priority, 100)
        )

        try:
            timestamp = (
                float(created_at)
                if created_at is not None
                else time.time()
            )
        except (TypeError, ValueError, OverflowError):
            timestamp = time.time()

        if not math.isfinite(timestamp) or timestamp <= 0:
            timestamp = time.time()

        self.created_at = timestamp
        self.done = bool(done)
        self.result = str(result or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "priority": self.priority,
            "created_at": self.created_at,
            "done": self.done,
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        if not isinstance(data, dict):
            raise TypeError("Invalid goal data")

        return cls(
            text=data.get("text", ""),
            source=data.get("source", "self"),
            priority=data.get("priority", 1),
            created_at=data.get("created_at"),
            done=data.get("done", False),
            result=data.get("result", "")
        )


class GoalStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(
            path
            if path is not None
            else Config.LIFE_DIR / "goals.json"
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.goals: List[Goal] = []
        self._lock = threading.RLock()

        self._load()

    def _max_goals(self) -> int:
        try:
            return max(
                1,
                int(Config.LIFE_MAX_GOALS)
            )
        except (TypeError, ValueError):
            return 10000

    def _normalize_text(self, text: str) -> str:
        return " ".join(
            text.casefold().replace("ё", "е").split()
        )

    def _trim_goals(self):
        limit = self._max_goals() * 3

        if len(self.goals) <= limit:
            return

        active = [
            goal for goal in self.goals
            if not goal.done
        ]

        completed = [
            goal for goal in self.goals
            if goal.done
        ]

        if len(active) >= limit:
            active.sort(
                key=lambda goal: (
                    -goal.priority,
                    goal.created_at
                )
            )

            self.goals = active[:limit]
            return

        remaining = limit - len(active)

        completed.sort(
            key=lambda goal: goal.created_at,
            reverse=True
        )

        self.goals = active + completed[:remaining]

    def _load(self) -> None:
        with self._lock:
            self.goals = []

            if not self.path.is_file():
                return

            try:
                data = json.loads(
                    self.path.read_text(
                        encoding="utf-8"
                    )
                )

                if not isinstance(data, dict):
                    return

                items = data.get("goals", [])

                if not isinstance(items, list):
                    return

                for item in items:
                    try:
                        goal = Goal.from_dict(item)
                    except (
                        TypeError,
                        ValueError,
                        OverflowError
                    ):
                        continue

                    if goal.text:
                        self.goals.append(goal)

                self._trim_goals()

            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError
            ):
                self.goals = []

    def save(self) -> None:
        with self._lock:
            self._trim_goals()

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            data = {
                "goals": [
                    goal.to_dict()
                    for goal in self.goals
                ]
            }

            temporary = None

            try:
                fd, filename = tempfile.mkstemp(
                    prefix=".goals_",
                    suffix=".tmp",
                    dir=str(self.path.parent)
                )

                temporary = Path(filename)

                with os.fdopen(
                    fd,
                    "w",
                    encoding="utf-8"
                ) as stream:
                    json.dump(
                        data,
                        stream,
                        ensure_ascii=False,
                        indent=2
                    )

                    stream.flush()
                    os.fsync(stream.fileno())

                os.replace(
                    temporary,
                    self.path
                )

            finally:
                if temporary is not None:
                    temporary.unlink(
                        missing_ok=True
                    )

    def active(self) -> List[Goal]:
        with self._lock:
            return [
                goal
                for goal in self.goals
                if not goal.done
            ]

    def add(
        self,
        text: str,
        source: str = "self",
        priority: int = 1
    ) -> Goal:
        if not isinstance(text, str):
            raise TypeError("Goal text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("Пустая цель")

        key = self._normalize_text(text)

        with self._lock:
            for goal in self.goals:
                if (
                    not goal.done
                    and self._normalize_text(goal.text) == key
                ):
                    return goal

            goal = Goal(
                text=text,
                source=source,
                priority=priority
            )

            self.goals.append(goal)
            self._trim_goals()
            self.save()

            return goal

    def pick_next(self) -> Optional[Goal]:
        with self._lock:
            active_goals = [
                goal
                for goal in self.goals
                if not goal.done
            ]

            if not active_goals:
                return None

            return min(
                active_goals,
                key=lambda goal: (
                    -goal.priority,
                    goal.created_at
                )
            )

    def complete(
        self,
        goal: Goal,
        result: str = ""
    ) -> None:
        with self._lock:
            if goal not in self.goals:
                return

            goal.done = True
            goal.result = str(result or "")
            self.save()

    def fail(
        self,
        goal: Goal,
        reason: str = ""
    ) -> None:
        with self._lock:
            if goal not in self.goals:
                return

            goal.result = f"FAIL: {reason}"
            self.save()
