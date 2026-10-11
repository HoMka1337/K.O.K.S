
import inspect
import logging
import threading
from typing import Any, Callable, Dict


class Executor:
    def __init__(self):
        self.tools = {}
        self.max_iterations = 10
        self._lock = threading.RLock()
        self._logger = logging.getLogger("koks.executor")

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid tool name")

        if not callable(func):
            raise TypeError("Tool must be callable")

        if not isinstance(description, str):
            description = str(description)

        with self._lock:
            self.tools[name] = {
                "func": func,
                "description": description
            }

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any]
    ) -> str:
        if not isinstance(tool_name, str):
            return "Некорректное имя инструмента"

        if not isinstance(args, dict):
            return "Некорректные аргументы инструмента"

        with self._lock:
            tool = self.tools.get(tool_name)

        if tool is None:
            return f"Инструмент {tool_name} не найден"

        func = tool["func"]

        try:
            signature = inspect.signature(func)

            signature.bind(**args)

        except (TypeError, ValueError) as error:
            return f"Некорректные аргументы: {error}"

        try:
            result = func(**args)

            if result is None:
                return ""

            if isinstance(result, str):
                return result

            return str(result)

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as error:
            self._logger.exception(
                "Tool execution failed: %s",
                tool_name
            )

            return (
                f"Ошибка выполнения инструмента "
                f"{tool_name}: "
                f"{type(error).__name__}: {error}"
            )

    def get_tools_description(self) -> str:
        with self._lock:
            tools = list(self.tools.items())

        result = []

        for name, tool in tools:
            description = tool.get(
                "description",
                ""
            )

            result.append(
                f"{name}: {description}"
            )

        return "\n".join(result)
