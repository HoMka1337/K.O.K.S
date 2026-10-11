import subprocess
import tempfile
from pathlib import Path
import ast

from config import Config
from tools.security import get_security


class CodeTools:
    @staticmethod
    def execute_python(code: str, timeout: int = None) -> str:
        timeout = timeout or Config.TOOLS_MAX_PYTHON_TIME
        sec = get_security()
        ok, reason = sec.check_code(code)
        if not ok:
            return reason
        try:
            sandbox = Config.SANDBOX_DIR
            sandbox.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=str(sandbox), encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name
            try:
                result = subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(sandbox),
                )
            finally:
                try:
                    Path(temp_path).unlink()
                except OSError:
                    pass
            sec.commit_code()
            output = (result.stdout or "") + (result.stderr or "")
            if len(output) > Config.TOOLS_MAX_OUTPUT_CHARS:
                output = output[: Config.TOOLS_MAX_OUTPUT_CHARS] + "...[обрезано]"
            if result.returncode == 0:
                return output if output else "Код выполнен успешно"
            return f"Ошибка выполнения:\n{output}"
        except subprocess.TimeoutExpired:
            return f"Превышено время выполнения ({timeout} сек)"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def analyze_python(code: str) -> str:
        try:
            tree = ast.parse(code)
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            imports = [
                n.names[0].name
                for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            result = "Анализ Python кода:\n"
            result += f"  Строк: {len(code.splitlines())}\n"
            result += f"  Символов: {len(code)}\n"
            result += f"  Функций: {len(funcs)}\n"
            result += f"  Классов: {len(classes)}\n"
            result += f"  Импортов: {len(set(imports))}\n"
            if funcs:
                result += f"  Функции: {', '.join(funcs[:10])}\n"
            if classes:
                result += f"  Классы: {', '.join(classes[:10])}\n"
            return result
        except SyntaxError as e:
            return f"Синтаксическая ошибка: {e}"
        except Exception as e:
            return f"Ошибка анализа: {e}"

    @staticmethod
    def format_python(code: str) -> str:
        try:
            tree = ast.parse(code)
            import astor
            formatted = astor.to_source(tree)
            return f"Отформатированный код:\n\n{formatted}"
        except Exception:
            lines = code.splitlines()
            result = []
            indent = 0
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    result.append("")
                    continue
                if stripped.endswith(":"):
                    result.append("    " * indent + stripped)
                    indent += 1
                elif stripped.startswith("return") or stripped.startswith("pass"):
                    result.append("    " * max(0, indent - 1) + stripped)
                else:
                    result.append("    " * indent + stripped)
            return "\n".join(result)