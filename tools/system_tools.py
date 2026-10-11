
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from config import Config
from tools.security import get_security


class SystemTools:
    @staticmethod
    def _limit_output(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        limit = max(
            1,
            int(getattr(Config, "TOOLS_MAX_OUTPUT_CHARS", 10000))
        )

        if len(text) > limit:
            return text[:limit] + "\n[Вывод обрезан]"

        return text

    @staticmethod
    def _parse_command(command: str):
        if not isinstance(command, str):
            raise ValueError("Команда должна быть строкой")

        command = command.strip()

        if not command:
            raise ValueError("Пустая команда")

        if len(command) > 4096:
            raise ValueError("Слишком длинная команда")

        if any(
            ord(character) < 32
            for character in command
        ):
            raise ValueError("Недопустимые символы")

        if re.search(r"[;&|<>`$]", command):
            raise ValueError("Недопустимые операторы")

        try:
            parts = shlex.split(
                command,
                posix=os.name != "nt"
            )
        except ValueError as error:
            raise ValueError(
                f"Ошибка разбора команды: {error}"
            ) from error

        if not parts:
            raise ValueError("Пустая команда")

        if os.name == "nt":
            parts = [
                part[1:-1]
                if len(part) >= 2
                and part[0] == part[-1]
                and part[0] in ("'", '"')
                else part
                for part in parts
            ]

        name = parts[0].casefold()
        args = parts[1:]

        allowed = {
            "dir",
            "ls",
            "type",
            "cat",
            "echo",
            "where",
            "which",
            "python",
            "pip"
        }

        if name not in allowed:
            raise ValueError("Команда не поддерживается")

        if name == "python" and args != ["--version"]:
            raise ValueError("Разрешено только python --version")

        if name == "pip":
            if not args:
                raise ValueError("Не указана команда pip")

            action = args[0].casefold()

            if action == "list":
                if len(args) != 1:
                    raise ValueError("Разрешено только pip list")

            elif action == "show":
                if len(args) != 2:
                    raise ValueError("Укажите один пакет")

                if not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                    args[1]
                ):
                    raise ValueError("Некорректное имя пакета")

            else:
                raise ValueError(
                    "Разрешены только pip list и pip show"
                )

        if name in {"dir", "ls", "type", "cat"}:
            if len(args) > 1:
                raise ValueError("Допускается только один путь")

            if args and args[0].startswith("-"):
                raise ValueError("Параметры не поддерживаются")

        if name in {"where", "which"}:
            if len(args) != 1:
                raise ValueError("Укажите имя программы")

            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
                args[0]
            ):
                raise ValueError("Некорректное имя программы")

        return name, args

    @staticmethod
    def _resolve_path(value: str = ".") -> Path:
        base = Path(Config.BASE_DIR).resolve()

        path = Path(value).expanduser()

        if not path.is_absolute():
            path = base / path

        resolved = path.resolve()

        if not (
            resolved == base
            or base in resolved.parents
        ):
            raise PermissionError(
                "Доступ за пределами проекта запрещён"
            )

        return resolved

    @staticmethod
    def _list_directory(args):
        target = SystemTools._resolve_path(
            args[0] if args else "."
        )

        if not target.exists():
            return f"Путь не найден: {target}"

        if not target.is_dir():
            return f"Это не папка: {target}"

        entries = sorted(
            target.iterdir(),
            key=lambda entry: (
                not entry.is_dir(),
                entry.name.casefold()
            )
        )

        result = []

        for entry in entries[:200]:
            try:
                resolved = entry.resolve()

                base = Path(Config.BASE_DIR).resolve()

                if not (
                    resolved == base
                    or base in resolved.parents
                ):
                    continue

                if entry.is_dir():
                    result.append(
                        f"[DIR] {entry.name}"
                    )
                elif entry.is_file():
                    size = entry.stat().st_size

                    result.append(
                        f"[FILE] {entry.name} ({size} байт)"
                    )

            except (OSError, PermissionError):
                continue

        if len(entries) > 200:
            result.append(
                f"Показаны первые 200 из {len(entries)}"
            )

        return "\n".join(result) or "Папка пуста"

    @staticmethod
    def _read_file(args):
        if len(args) != 1:
            return "Укажите файл"

        target = SystemTools._resolve_path(args[0])

        if not target.is_file():
            return f"Файл не найден: {target}"

        allowed_extensions = getattr(
            Config,
            "ALLOWED_EXTENSIONS",
            (".txt", ".md", ".json", ".py", ".log")
        )

        if target.suffix.casefold() not in allowed_extensions:
            return "Чтение этого типа файла запрещено"

        limit = max(
            1,
            int(getattr(Config, "TOOLS_MAX_OUTPUT_CHARS", 10000))
        )

        with target.open(
            "r",
            encoding="utf-8",
            errors="replace"
        ) as stream:
            content = stream.read(limit + 1)

        return SystemTools._limit_output(content)

    @staticmethod
    def _find_program(args):
        name = args[0]
        executable = shutil.which(name)

        if not executable:
            return f"Программа не найдена: {name}"

        return executable

    @staticmethod
    def _run_pip(args):
        command = [
            sys.executable,
            "-m",
            "pip"
        ] + args

        result = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(
                1,
                int(Config.TOOLS_MAX_EXEC_TIME)
            ),
            cwd=str(Config.BASE_DIR),
            stdin=subprocess.DEVNULL,
            check=False
        )

        output = (
            (result.stdout or "")
            + (result.stderr or "")
        ).strip()

        if result.returncode != 0:
            output = (
                f"Код завершения: {result.returncode}\n"
                + output
            )

        return output or "Нет вывода"

    @staticmethod
    def execute_command(command: str) -> str:
        if not getattr(
            Config,
            "TOOLS_ENABLE_SYSTEM",
            True
        ):
            return "Системные инструменты отключены"

        if not getattr(
            Config,
            "ALLOW_COMMAND_EXECUTION",
            False
        ):
            return "Выполнение системных команд отключено"

        security = get_security()

        ok, reason = security.check_command(command)

        if not ok:
            return reason

        try:
            name, args = SystemTools._parse_command(
                command
            )

            if name in {"dir", "ls"}:
                output = SystemTools._list_directory(args)

            elif name in {"type", "cat"}:
                output = SystemTools._read_file(args)

            elif name == "echo":
                output = " ".join(args)

            elif name in {"where", "which"}:
                output = SystemTools._find_program(args)

            elif name == "python":
                output = (
                    f"Python {platform.python_version()}"
                )

            elif name == "pip":
                output = SystemTools._run_pip(args)

            else:
                return "Команда не поддерживается"

            security.commit_command()

            return SystemTools._limit_output(
                output or "Выполнено"
            )

        except subprocess.TimeoutExpired:
            return (
                "Превышено время выполнения "
                f"({Config.TOOLS_MAX_EXEC_TIME} сек)"
            )

        except (
            OSError,
            ValueError,
            TypeError,
            PermissionError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def get_system_info() -> str:
        info = {
            "OS": platform.system(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Python": platform.python_version(),
            "Architecture": platform.architecture()[0],
            "Executable": sys.executable
        }

        return "\n".join(
            f"{key}: {value}"
            for key, value in info.items()
        )
