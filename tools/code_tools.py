
import ast
from typing import Optional

from config import Config


class CodeTools:
    @staticmethod
    def execute_python(
        code: str,
        timeout: Optional[int] = None
    ) -> str:
        if not isinstance(code, str):
            return "Ошибка: код должен быть строкой"

        if not code.strip():
            return "Ошибка: пустой код"

        if not getattr(Config, "TOOLS_ENABLE_CODE", True):
            return "Инструменты Python отключены"

        if not getattr(Config, "ALLOW_CODE_EXECUTION", False):
            return "Выполнение произвольного Python отключено"

        try:
            ast.parse(code)
        except SyntaxError as error:
            return (
                f"Синтаксическая ошибка: "
                f"строка {error.lineno}, "
                f"позиция {error.offset}: "
                f"{error.msg}"
            )
        except (ValueError, TypeError) as error:
            return f"Ошибка анализа: {error}"

        return (
            "Выполнение произвольного Python недоступно: "
            "изолированная среда выполнения не настроена"
        )

    @staticmethod
    def analyze_python(code: str) -> str:
        if not isinstance(code, str):
            return "Ошибка: код должен быть строкой"

        if not code.strip():
            return "Ошибка: пустой код"

        try:
            tree = ast.parse(code)

            functions = []
            async_functions = []
            classes = []
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    async_functions.append(node.name)

                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)

                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""

                    if node.level:
                        module = "." * node.level + module

                    if module:
                        imports.add(module)

                    else:
                        for alias in node.names:
                            imports.add(alias.name)

            all_functions = functions + async_functions

            result = [
                "Анализ Python кода:",
                f"Строк: {len(code.splitlines())}",
                f"Символов: {len(code)}",
                f"Функций: {len(all_functions)}",
                f"Асинхронных функций: {len(async_functions)}",
                f"Классов: {len(classes)}",
                f"Импортов: {len(imports)}"
            ]

            if all_functions:
                result.append(
                    "Функции: "
                    + ", ".join(all_functions[:20])
                )

            if classes:
                result.append(
                    "Классы: "
                    + ", ".join(classes[:20])
                )

            if imports:
                result.append(
                    "Импорты: "
                    + ", ".join(sorted(imports)[:20])
                )

            return "\n".join(result)

        except SyntaxError as error:
            return (
                f"Синтаксическая ошибка: "
                f"строка {error.lineno}, "
                f"позиция {error.offset}: "
                f"{error.msg}"
            )

        except (TypeError, ValueError, RecursionError) as error:
            return f"Ошибка анализа: {error}"

    @staticmethod
    def format_python(code: str) -> str:
        if not isinstance(code, str):
            return "Ошибка: код должен быть строкой"

        if not code.strip():
            return "Ошибка: пустой код"

        try:
            tree = ast.parse(code)
            formatted = ast.unparse(tree)

            if not formatted:
                return ""

            return formatted.rstrip() + "\n"

        except SyntaxError as error:
            return (
                f"Синтаксическая ошибка: "
                f"строка {error.lineno}, "
                f"позиция {error.offset}: "
                f"{error.msg}"
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
            RecursionError
        ) as error:
            return f"Ошибка форматирования: {error}"
