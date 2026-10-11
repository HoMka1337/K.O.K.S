import json
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
        result: str = "",
    ):
        self.text = text
        self.source = source
        self.priority = priority
        self.created_at = created_at or time.time()
        self.done = done
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "priority": self.priority,
            "created_at": self.created_at,
            "done": self.done,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            text=data.get("text", ""),
            source=data.get("source", "self"),
            priority=int(data.get("priority", 1)),
            created_at=data.get("created_at"),
            done=bool(data.get("done", False)),
            result=data.get("result", ""),
        )


class GoalStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else (Config.LIFE_DIR / "goals.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.goals: List[Goal] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            self.goals = [Goal.from_dict(item) for item in data.get("goals", [])]
        except (OSError, json.JSONDecodeError):
            self.goals = []

    def save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(
                    {"goals": [g.to_dict() for g in self.goals]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
        except OSError:
            pass

    def active(self) -> List[Goal]:
        return [g for g in self.goals if not g.done]

    def add(self, text: str, source: str = "self", priority: int = 1) -> Goal:
        text = text.strip()
        if not text:
            raise ValueError("Пустая цель")
        existing = {g.text for g in self.goals if not g.done}
        if text in existing:
            return next(g for g in self.goals if g.text == text and not g.done)
        goal = Goal(text=text, source=source, priority=priority)
        self.goals.append(goal)
        if len(self.goals) > Config.LIFE_MAX_GOALS * 3:
            self.goals = self.goals[-Config.LIFE_MAX_GOALS * 3:]
        self.save()
        return goal

    def pick_next(self) -> Optional[Goal]:
        active = self.active()
        if not active:
            return None
        active.sort(key=lambda g: (-g.priority, g.created_at))
        return active[0]

    def complete(self, goal: Goal, result: str = "") -> None:
        goal.done = True
        goal.result = result
        self.save()

    def fail(self, goal: Goal, reason: str = "") -> None:
        goal.result = f"FAIL: {reason}"
        self.save()