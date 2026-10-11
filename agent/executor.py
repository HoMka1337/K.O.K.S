from typing import Dict, Any, Callable

class Executor:
    def __init__(self):
        self.tools = {}
        self.max_iterations = 10
    
    def register_tool(self, name: str, func: Callable, description: str):
        self.tools[name] = {
            'func': func,
            'description': description
        }
    
    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name not in self.tools:
            return f"Инструмент {tool_name} не найден"
        try:
            result = self.tools[tool_name]['func'](**args)
            return str(result)
        except Exception as e:
            return f"Ошибка выполнения: {e}"
    
    def get_tools_description(self) -> str:
        result = []
        for name, tool in self.tools.items():
            result.append(f"  • {name}: {tool['description']}")
        return "\n".join(result)