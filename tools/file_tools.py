import shutil
from pathlib import Path

from config import Config
from tools.security import get_security


class FileTools:
    @staticmethod
    def _guard_read(path: str) -> str:
        if not Config.can_read(path):
            return f"Чтение запрещено: {path}"
        return ""

    @staticmethod
    def _guard_write(path: str) -> str:
        sec = get_security()
        ok, reason = sec.check_write(path)
        if not ok:
            return reason
        return ""

    @staticmethod
    def _guard_delete(path: str) -> str:
        sec = get_security()
        ok, reason = sec.check_delete(path)
        if not ok:
            return reason
        return ""

    @staticmethod
    def read_file(path: str) -> str:
        err = FileTools._guard_read(path)
        if err:
            return err
        try:
            p = Path(path)
            if not p.exists():
                return f"Файл не найден: {path}"
            if p.stat().st_size > 5 * 1024 * 1024:
                return f"Файл слишком большой: {path}"
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        err = FileTools._guard_write(path)
        if err:
            return err
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            get_security().commit_write()
            return f"Файл сохранён: {path}"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def append_file(path: str, content: str) -> str:
        err = FileTools._guard_write(path)
        if err:
            return err
        try:
            p = Path(path)
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            get_security().commit_write()
            return f"Добавлено в файл: {path}"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def list_files(path: str = ".") -> str:
        if not Config.can_read(path):
            return f"Чтение запрещено: {path}"
        try:
            p = Path(path)
            if not p.exists():
                return f"Путь не найден: {path}"
            items = []
            for item in sorted(p.iterdir()):
                if item.is_dir():
                    items.append(f"[ПАПКА] {item.name}/")
                else:
                    items.append(f"[ФАЙЛ] {item.name} ({item.stat().st_size} байт)")
            return "\n".join(items) if items else "Папка пуста"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def delete_file(path: str) -> str:
        err = FileTools._guard_delete(path)
        if err:
            return err
        try:
            p = Path(path)
            if not p.exists():
                return f"Путь не найден: {path}"
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            get_security().commit_delete()
            return f"Удалено: {path}"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def copy_file(source: str, dest: str) -> str:
        if not Config.can_read(source):
            return f"Чтение запрещено: {source}"
        err = FileTools._guard_write(dest)
        if err:
            return err
        try:
            src = Path(source)
            dst = Path(dest)
            if not src.exists():
                return f"Источник не найден: {source}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            get_security().commit_write()
            return f"Скопировано: {source} -> {dest}"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def move_file(source: str, dest: str) -> str:
        if not Config.can_read(source):
            return f"Чтение запрещено: {source}"
        err = FileTools._guard_write(dest)
        if err:
            return err
        try:
            src = Path(source)
            dst = Path(dest)
            if not src.exists():
                return f"Источник не найден: {source}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
            get_security().commit_write()
            return f"Перемещено: {source} -> {dest}"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def create_folder(path: str) -> str:
        err = FileTools._guard_write(path)
        if err:
            return err
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            get_security().commit_write()
            return f"Папка создана: {path}"
        except Exception as e:
            return f"Ошибка: {e}"