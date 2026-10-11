
import os
import shutil
import tempfile
import threading
from pathlib import Path

from config import Config
from tools.security import get_security


class FileTools:
    _lock = threading.RLock()
    _max_file_size = 5 * 1024 * 1024
    _max_list_items = 200

    @staticmethod
    def _resolve_path(path):
        if not isinstance(path, (str, Path)):
            raise ValueError("Некорректный путь")

        path = Path(path).expanduser()

        return path.resolve(strict=False)

    @staticmethod
    def _guard_read(path: str) -> str:
        if not Config.can_read(path):
            return f"Чтение запрещено: {path}"

        return ""

    @staticmethod
    def _guard_write(path: str) -> str:
        security = get_security()
        allowed, reason = security.check_write(path)

        if not allowed:
            return reason

        return ""

    @staticmethod
    def _guard_delete(path: str) -> str:
        security = get_security()
        allowed, reason = security.check_delete(path)

        if not allowed:
            return reason

        return ""

    @staticmethod
    def _validate_content(content):
        if not isinstance(content, str):
            raise TypeError("Содержимое должно быть строкой")

        size = len(content.encode("utf-8"))

        if size > FileTools._max_file_size:
            raise ValueError("Содержимое слишком большое")

        return size

    @staticmethod
    def read_file(path: str) -> str:
        try:
            target = FileTools._resolve_path(path)

            error = FileTools._guard_read(str(target))

            if error:
                return error

            if not target.is_file():
                return f"Файл не найден: {path}"

            if target.stat().st_size > FileTools._max_file_size:
                return f"Файл слишком большой: {path}"

            with target.open(
                "r",
                encoding="utf-8",
                errors="replace"
            ) as stream:
                content = stream.read(
                    FileTools._max_file_size + 1
                )

            return content

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            target = FileTools._resolve_path(path)
            FileTools._validate_content(content)

            error = FileTools._guard_write(str(target))

            if error:
                return error

            if target.is_dir():
                return f"Путь является папкой: {path}"

            with FileTools._lock:
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                fd, temporary_name = tempfile.mkstemp(
                    prefix=".write_",
                    suffix=".tmp",
                    dir=str(target.parent)
                )

                temporary = Path(temporary_name)

                try:
                    with os.fdopen(
                        fd,
                        "w",
                        encoding="utf-8"
                    ) as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())

                    if not Config.can_write(str(target)):
                        return f"Запись запрещена: {path}"

                    os.replace(temporary, target)

                    get_security().commit_write()

                finally:
                    temporary.unlink(
                        missing_ok=True
                    )

            return f"Файл сохранён: {path}"

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def append_file(path: str, content: str) -> str:
        try:
            target = FileTools._resolve_path(path)
            content_size = FileTools._validate_content(content)

            error = FileTools._guard_write(str(target))

            if error:
                return error

            with FileTools._lock:
                if target.exists() and not target.is_file():
                    return f"Это не файл: {path}"

                if target.exists():
                    current_size = target.stat().st_size
                else:
                    current_size = 0

                if (
                    current_size + content_size
                    > FileTools._max_file_size
                ):
                    return f"Файл слишком большой: {path}"

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                if not Config.can_write(str(target)):
                    return f"Запись запрещена: {path}"

                with target.open(
                    "a",
                    encoding="utf-8"
                ) as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())

                get_security().commit_write()

            return f"Добавлено в файл: {path}"

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def list_files(path: str = ".") -> str:
        try:
            target = FileTools._resolve_path(path)

            error = FileTools._guard_read(str(target))

            if error:
                return error

            if not target.exists():
                return f"Путь не найден: {path}"

            if not target.is_dir():
                return f"Это не папка: {path}"

            items = []

            with FileTools._lock:
                entries = sorted(
                    target.iterdir(),
                    key=lambda item: item.name.casefold()
                )

                for entry in entries:
                    if len(items) >= FileTools._max_list_items:
                        break

                    try:
                        if not Config.can_read(str(entry)):
                            continue

                        if entry.is_symlink():
                            items.append(
                                f"[ССЫЛКА] {entry.name}"
                            )

                        elif entry.is_dir():
                            items.append(
                                f"[ПАПКА] {entry.name}/"
                            )

                        elif entry.is_file():
                            size = entry.stat().st_size

                            items.append(
                                f"[ФАЙЛ] {entry.name} "
                                f"({size} байт)"
                            )

                    except (
                        OSError,
                        ValueError,
                        RuntimeError
                    ):
                        continue

            if not items:
                return "Папка пуста"

            return "\n".join(items)

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def delete_file(path: str) -> str:
        try:
            target = FileTools._resolve_path(path)

            error = FileTools._guard_delete(str(target))

            if error:
                return error

            with FileTools._lock:
                if not target.exists():
                    return f"Путь не найден: {path}"

                if not Config.can_delete(str(target)):
                    return f"Удаление запрещено: {path}"

                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

                get_security().commit_delete()

            return f"Удалено: {path}"

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def copy_file(source: str, dest: str) -> str:
        try:
            src = FileTools._resolve_path(source)
            dst = FileTools._resolve_path(dest)

            error = FileTools._guard_read(str(src))

            if error:
                return error

            error = FileTools._guard_write(str(dst))

            if error:
                return error

            if src == dst:
                return "Источник и назначение совпадают"

            with FileTools._lock:
                if not src.is_file():
                    return f"Источник не найден: {source}"

                if dst.is_dir():
                    return f"Назначение является папкой: {dest}"

                if src.stat().st_size > FileTools._max_file_size:
                    return f"Файл слишком большой: {source}"

                dst.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                if not Config.can_write(str(dst)):
                    return f"Запись запрещена: {dest}"

                fd, temporary_name = tempfile.mkstemp(
                    prefix=".copy_",
                    suffix=".tmp",
                    dir=str(dst.parent)
                )

                os.close(fd)
                temporary = Path(temporary_name)

                try:
                    shutil.copyfile(src, temporary)
                    os.replace(temporary, dst)
                    get_security().commit_write()

                finally:
                    temporary.unlink(
                        missing_ok=True
                    )

            return f"Скопировано: {source} -> {dest}"

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def move_file(source: str, dest: str) -> str:
        try:
            src = FileTools._resolve_path(source)
            dst = FileTools._resolve_path(dest)

            error = FileTools._guard_delete(str(src))

            if error:
                return error

            error = FileTools._guard_write(str(dst))

            if error:
                return error

            if src == dst:
                return "Источник и назначение совпадают"

            with FileTools._lock:
                if not src.exists():
                    return f"Источник не найден: {source}"

                if dst.exists():
                    return f"Назначение уже существует: {dest}"

                if src.is_dir() and src in dst.parents:
                    return "Нельзя переместить папку внутрь себя"

                dst.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                if not Config.can_delete(str(src)):
                    return f"Перемещение запрещено: {source}"

                if not Config.can_write(str(dst)):
                    return f"Перемещение запрещено: {dest}"

                shutil.move(
                    str(src),
                    str(dst)
                )

                security = get_security()
                security.commit_write()
                security.commit_delete()

            return f"Перемещено: {source} -> {dest}"

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"

    @staticmethod
    def create_folder(path: str) -> str:
        try:
            target = FileTools._resolve_path(path)

            error = FileTools._guard_write(str(target))

            if error:
                return error

            with FileTools._lock:
                if target.exists() and not target.is_dir():
                    return f"Путь занят файлом: {path}"

                if not Config.can_write(str(target)):
                    return f"Создание запрещено: {path}"

                target.mkdir(
                    parents=True,
                    exist_ok=True
                )

                get_security().commit_write()

            return f"Папка создана: {path}"

        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError
        ) as error:
            return f"Ошибка: {error}"
