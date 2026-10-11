import time
import json
import random
import string
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import Config


class SecurityManager:
    def __init__(self):
        self.writes = 0
        self.deletes = 0
        self.commands = 0
        self.code_runs = 0
        self.network_calls = 0
        self.suspicious = 0
        self.locked = False
        self.recent_ops = deque(maxlen=Config.MAX_OPERATIONS_PER_MINUTE * 2)
        self.pending_confirmations = {}
        self._load_state()

    def _load_state(self):
        path = Config.SECURITY_STATE_FILE
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.writes = int(data.get("writes", 0))
            self.deletes = int(data.get("deletes", 0))
            self.commands = int(data.get("commands", 0))
            self.code_runs = int(data.get("code_runs", 0))
            self.suspicious = int(data.get("suspicious", 0))
            self.locked = bool(data.get("locked", False))
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def _save_state(self):
        try:
            Config.SECURITY_STATE_FILE.write_text(
                json.dumps(
                    {
                        "writes": self.writes,
                        "deletes": self.deletes,
                        "commands": self.commands,
                        "code_runs": self.code_runs,
                        "suspicious": self.suspicious,
                        "locked": self.locked,
                        "updated": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _rate_ok(self) -> bool:
        now = time.time()
        cutoff = now - 60.0
        while self.recent_ops and self.recent_ops[0] < cutoff:
            self.recent_ops.popleft()
        if len(self.recent_ops) >= Config.MAX_OPERATIONS_PER_MINUTE:
            return False
        self.recent_ops.append(now)
        return True

    def _flag_suspicious(self, reason: str):
        self.suspicious += 1
        Config._log_security(f"SUSPICIOUS: {reason}", "WARN")
        if Config.LOCK_AFTER_SUSPICIOUS and self.suspicious >= Config.SUSPICIOUS_ACTIONS_TO_LOCK:
            self.locked = True
            Config._log_security("LOCKED after suspicious activity", "ERROR")
        self._save_state()

    def check_write(self, path) -> tuple:
        if self.locked:
            return False, "Заблокировано после подозрительной активности"
        if not Config.can_write(path):
            self._flag_suspicious(f"write denied: {path}")
            return False, f"Запись запрещена: {path}"
        if self.writes >= Config.MAX_WRITES_PER_SESSION:
            return False, "Достигнут лимит записей за сессию"
        if not self._rate_ok():
            self._flag_suspicious("rate limit")
            return False, "Слишком много операций в минуту"
        return True, ""

    def check_delete(self, path) -> tuple:
        if self.locked:
            return False, "Заблокировано после подозрительной активности"
        if not Config.can_delete(path):
            self._flag_suspicious(f"delete denied: {path}")
            return False, f"Удаление запрещено: {path}"
        if self.deletes >= Config.MAX_DELETES_PER_SESSION:
            return False, "Достигнут лимит удалений за сессию"
        if not self._rate_ok():
            self._flag_suspicious("rate limit delete")
            return False, "Слишком много операций в минуту"
        return True, ""

    def check_command(self, command: str) -> tuple:
        if self.locked:
            return False, "Заблокировано после подозрительной активности"
        if not Config.can_execute_command(command):
            self._flag_suspicious(f"command denied: {command[:80]}")
            return False, f"Команда запрещена: {command[:80]}"
        if self.commands >= Config.MAX_COMMANDS_PER_SESSION:
            return False, "Достигнут лимит команд за сессию"
        if not self._rate_ok():
            self._flag_suspicious("rate limit command")
            return False, "Слишком много операций в минуту"
        return True, ""

    def check_code(self, code: str) -> tuple:
        if self.locked:
            return False, "Заблокировано после подозрительной активности"
        if not Config.can_execute_code(code):
            self._flag_suspicious(f"code denied: {code[:80]}")
            return False, "Код содержит запрещённые вызовы"
        if self.code_runs >= Config.MAX_CODE_RUNS_PER_SESSION:
            return False, "Достигнут лимит запусков кода за сессию"
        if not self._rate_ok():
            self._flag_suspicious("rate limit code")
            return False, "Слишком много операций в минуту"
        return True, ""

    def check_network(self) -> tuple:
        if self.locked:
            return False, "Заблокировано после подозрительной активности"
        if not Config.can_use_network():
            return False, "Сеть запрещена"
        self.network_calls += 1
        return True, ""

    def commit_write(self):
        self.writes += 1
        self._save_state()

    def commit_delete(self):
        self.deletes += 1
        self._save_state()

    def commit_command(self):
        self.commands += 1
        self._save_state()

    def commit_code(self):
        self.code_runs += 1
        self._save_state()

    def generate_confirmation_code(self, action: str) -> str:
        code = "".join(random.choices(string.digits, k=Config.CONFIRMATION_CODE_LENGTH))
        self.pending_confirmations[code] = {
            "action": action,
            "created": time.time(),
        }
        Config._log_security(f"confirmation requested for: {action[:80]}")
        return code

    def verify_confirmation_code(self, code: str, max_age_sec: int = 120) -> bool:
        entry = self.pending_confirmations.pop(code, None)
        if entry is None:
            return False
        age = time.time() - entry["created"]
        if age > max_age_sec:
            return False
        return True

    def clear_pending(self):
        self.pending_confirmations.clear()

    def lock(self, reason: str = "manual"):
        self.locked = True
        Config._log_security(f"LOCKED: {reason}", "ERROR")
        self._save_state()

    def unlock(self):
        self.locked = False
        self.suspicious = 0
        Config._log_security("UNLOCKED")
        self._save_state()

    def report(self) -> str:
        return (
            f"SecurityManager:\n"
            f"  locked: {self.locked}\n"
            f"  writes: {self.writes}/{Config.MAX_WRITES_PER_SESSION}\n"
            f"  deletes: {self.deletes}/{Config.MAX_DELETES_PER_SESSION}\n"
            f"  commands: {self.commands}/{Config.MAX_COMMANDS_PER_SESSION}\n"
            f"  code_runs: {self.code_runs}/{Config.MAX_CODE_RUNS_PER_SESSION}\n"
            f"  network_calls: {self.network_calls}\n"
            f"  suspicious: {self.suspicious}"
        )


_security_manager: Optional[SecurityManager] = None


def get_security() -> SecurityManager:
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager