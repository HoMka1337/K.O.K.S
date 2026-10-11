import subprocess
import platform

from config import Config
from tools.security import get_security


class SystemTools:
    @staticmethod
    def execute_command(command: str) -> str:
        sec = get_security()
        ok, reason = sec.check_command(command)
        if not ok:
            return reason
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=Config.TOOLS_MAX_EXEC_TIME,
                env={"PATH": ""},
                cwd=str(Config.BASE_DIR),
            )
            sec.commit_command()
            output = (result.stdout or "") + (result.stderr or "")
            if len(output) > Config.TOOLS_MAX_OUTPUT_CHARS:
                output = output[:Config.TOOLS_MAX_OUTPUT_CHARS] + "...[обрезано]"
            return output if output else "Выполнено"
        except subprocess.TimeoutExpired:
            return f"Превышено время выполнения ({Config.TOOLS_MAX_EXEC_TIME} сек)"
        except Exception as e:
            return f"Ошибка: {e}"

    @staticmethod
    def get_system_info() -> str:
        info = {
            "OS": platform.system(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Python": platform.python_version(),
        }
        return "\n".join(f"{k}: {v}" for k, v in info.items())