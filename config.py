import os
import platform
import multiprocessing
import shutil
from pathlib import Path
from datetime import datetime
import torch


def _bytes_to_gb(value):
    return value / (1024 ** 3)


def _bytes_to_mb(value):
    return value / (1024 ** 2)


def detect_device():
    info = {}

    if torch.cuda.is_available():
        device = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_memory = _bytes_to_gb(props.total_memory)

        info["gpu_count"] = gpu_count
        info["gpu_name"] = gpu_name
        info["gpu_memory_gb"] = round(gpu_memory, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_multi_processor_count"] = props.multi_processor_count

        print(f"NVIDIA GPU найден: {gpu_name}")
        print(f"Видеопамять: {gpu_memory:.1f} GB")
        if gpu_count > 1:
            print(f"Обнаружено GPU: {gpu_count}")
        print(f"Compute capability: {info['gpu_compute_capability']}")

    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
        info["xpu_name"] = torch.xpu.get_device_name(0)
        print(f"Intel GPU найден: {info['xpu_name']}")

    elif hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        device = "mps"
        info["mps"] = True
        print("Apple MPS найден (Metal Performance Shaders)")

    else:
        device = "cpu"
        info["cpu"] = True
        print("Используется CPU")

    info["device"] = device
    return device, info


DEVICE, DEVICE_INFO = detect_device()


def detect_system_info():
    total_ram = 0
    available_ram = 0
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram = _bytes_to_gb(vm.total)
        available_ram = _bytes_to_gb(vm.available)
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": multiprocessing.cpu_count(),
        "hostname": platform.node(),
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "cwd": str(Path.cwd()),
        "total_ram_gb": round(total_ram, 2) if total_ram else None,
        "available_ram_gb": round(available_ram, 2) if available_ram else None,
    }


SYSTEM_INFO = detect_system_info()


class Config:
    VOCAB_SIZE = 24000
    D_MODEL = 320
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 1280
    MAX_LEN = 512
    DROPOUT = 0.1

    BATCH_SIZE = 1
    TRAIN_BATCH_SIZE = 16
    TRAIN_MAX_LEN = 192
    LEARNING_RATE = 3e-4
    EPOCHS = 30
    GRAD_CLIP = 1.0
    VALIDATION_SPLIT = 0.1
    WARMUP_STEPS = 200
    WEIGHT_DECAY = 0.01
    ADAM_BETAS = (0.9, 0.95)
    ADAM_EPS = 1e-8
    LR_SCHEDULER = "cosine"
    SAVE_BEST_ONLY = True
    SAVE_EVERY_EPOCH = True
    LOG_EVERY_N_STEPS = 10
    EARLY_STOPPING_PATIENCE = 5

    MAX_VRAM_GB = 4.5
    RESERVED_VRAM_GB = 1.5
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_GRADIENT_CHECKPOINTING = False
    EMPTY_CACHE_EVERY_N_STEPS = 50
    MAX_CACHED_TEXTS = 2000
    MAX_CACHED_TOKENS = 500_000
    MAX_RAM_GB = 5.0
    RESERVED_RAM_GB = 1.5

    TEMPERATURE = 0.85
    TOP_K_GEN = 50
    TOP_P_GEN = 0.95
    REPETITION_PENALTY = 1.15
    MAX_NEW_TOKENS = 150
    MIN_NEW_TOKENS = 4

    TOKENIZER_MIN_FREQ = 2
    TOKENIZER_MAX_VOCAB = 24000
    TOKENIZER_SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]

    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    LIFE_DIR = BASE_DIR / "life" / "state"
    CACHE_DIR = BASE_DIR / "cache"
    TMP_DIR = BASE_DIR / "tmp"
    WEB_DIR = BASE_DIR / "web"
    TESTS_DIR = BASE_DIR / "tests"
    BACKUP_DIR = BASE_DIR / "backups"
    SANDBOX_DIR = BASE_DIR / "sandbox"
    QUARANTINE_DIR = BASE_DIR / "quarantine"

    LOG_FILE = LOGS_DIR / "agent.log"
    ERROR_LOG_FILE = LOGS_DIR / "errors.log"
    SECURITY_LOG_FILE = LOGS_DIR / "security.log"
    MEMORY_FILE = BASE_DIR / "memory.json"
    KNOWLEDGE_FILE = BASE_DIR / "knowledge.json"
    DIALOGS_FILE = DATA_DIR / "dialogs.txt"
    TEACHER_DIALOGS_FILE = DATA_DIR / "teacher_dialogs.txt"
    LEARNED_COMMANDS_FILE = DATA_DIR / "learned_commands.txt"
    TOKENIZER_FILE = BASE_DIR / "tokenizer.json"
    MODEL_FILE = MODELS_DIR / "KoKs_model.pt"
    JOURNAL_FILE = LIFE_DIR / "journal.jsonl"
    GOALS_FILE = LIFE_DIR / "goals.json"
    CONFIG_SNAPSHOT = BASE_DIR / "config_snapshot.json"
    SECURITY_STATE_FILE = BASE_DIR / "security_state.json"

    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_MAX_BYTES = 5 * 1024 * 1024
    LOG_BACKUP_COUNT = 3
    LOG_TO_CONSOLE = True
    LOG_TO_FILE = True
    LOG_JSON = False

    DEVICE = DEVICE
    DEVICE_INFO = DEVICE_INFO
    SYSTEM_INFO = SYSTEM_INFO
    USE_AMP = DEVICE == "cuda"
    AMP_DTYPE = "bf16"
    PIN_MEMORY = DEVICE == "cuda"
    NUM_WORKERS = 0
    CUDA_ALLOC_CONF = "max_split_size_mb:256"

    MEMORY_MAX_SIZE = 60
    MEMORY_CONTEXT_LIMIT = 8
    KNOWLEDGE_MAX_ENTRIES = 10000
    KNOWLEDGE_MAX_CONTENT = 4000
    KNOWLEDGE_SEARCH_LIMIT = 1
    KNOWLEDGE_MIN_SCORE = 0.4

    SEMANTICS_MIN_SCORE = 0.5
    SEMANTICS_TYPE_BONUS = 1.3
    SEMANTICS_MAX_ANSWER = 600
    SEMANTICS_MIN_ANSWER_LETTERS = 40
    SEMANTICS_USE_SYNONYMS = True
    SEMANTICS_USE_STEMMING = True

    DIALOG_MIN_QUESTION_LEN = 3
    DIALOG_MIN_ANSWER_LEN = 10
    DIALOG_MAX_QUESTION_LEN = 200
    DIALOG_MAX_ANSWER_LEN = 2000
    DIALOGS_MAX_PAIRS = 10000

    WEB_TIMEOUT_SEC = 15
    WEB_MAX_PAGE_BYTES = 500_000
    WEB_MAX_PAGE_CHARS = 30000
    WEB_SEARCH_LIMIT = 5
    WEB_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    WEB_ACCEPT_LANGUAGE = "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"
    WEB_ACCEPT_ENCODING = "identity"
    WEB_RETRY_COUNT = 2
    WEB_RETRY_DELAY_SEC = 1.5
    WEB_ENCODE_CYRILLIC = True
    WEB_DECODE_GZIP = True
    WEB_DECODE_DEFLATE = True

    TOOLS_MAX_EXEC_TIME = 30
    TOOLS_MAX_PYTHON_TIME = 10
    TOOLS_MAX_OUTPUT_CHARS = 10000
    TOOLS_ENABLE_SYSTEM = True
    TOOLS_ENABLE_CODE = True
    TOOLS_ENABLE_WEB = True
    TOOLS_ENABLE_FILES = True
    RAG_ENABLED = False

    SAFE_MODE = True
    CONFIRM_DANGEROUS = True
    REQUIRE_CONFIRMATION_CODE = True
    CONFIRMATION_CODE_LENGTH = 6
    ALLOW_DANGEROUS_ACTIONS = False
    READ_ONLY_MODE = False
    SANDBOX_MODE = True
    ALLOW_WRITE_OUTSIDE_BASE = False
    ALLOW_DELETE_ANY = False
    ALLOW_COMMAND_EXECUTION = False
    ALLOW_CODE_EXECUTION = False
    ALLOW_NETWORK = True
    ALLOW_EXTERNAL_PATHS = False
    MAX_OPERATIONS_PER_MINUTE = 60
    MAX_DELETES_PER_SESSION = 50
    MAX_WRITES_PER_SESSION = 500
    MAX_COMMANDS_PER_SESSION = 20
    MAX_CODE_RUNS_PER_SESSION = 100
    LOCK_AFTER_SUSPICIOUS = True
    SUSPICIOUS_ACTIONS_TO_LOCK = 5

    ALLOWED_EXTENSIONS = (
        ".txt", ".md", ".json", ".jsonl", ".yaml", ".yml",
        ".py", ".js", ".ts", ".html", ".css", ".csv", ".log",
        ".ini", ".cfg", ".conf", ".toml",
    )

    FORBIDDEN_EXTENSIONS = (
        ".exe", ".bat", ".cmd", ".com", ".msi", ".scr", ".pif",
        ".dll", ".sys", ".drv", ".vbs", ".vbe", ".js", ".jse",
        ".wsf", ".wsh", ".ps1", ".psm1", ".reg", ".lnk",
    )

    FORBIDDEN_PATHS = (
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
        "C:\\Recovery",
        "C:\\$Recycle.Bin",
        "C:\\System Volume Information",
        "C:\\Users\\Default",
        "C:\\Users\\Public",
        "C:\\Users\\All Users",
        "C:\\PerfLogs",
        "C:\\Boot",
        "C:\\EFI",
        "C:\\MSOCache",
        "C:\\Intel",
        "C:\\AMD",
        "C:\\NVIDIA",
        "/etc", "/usr", "/bin", "/sbin", "/boot", "/root",
        "/var", "/opt", "/proc", "/sys", "/dev",
    )

    FORBIDDEN_PATH_PARTS = (
        "system32", "syswow64", "winsxs", "windows",
        "program files", "programdata",
        "appdata\\local\\microsoft",
        "appdata\\roaming\\microsoft",
        "system volume information",
        "recycle.bin", "$recycle.bin",
        "boot", "efi", "recovery",
        "microsoft\\windows", "microsoft\\crypto",
        "microsoft\\protect",
        ".ssh", ".gnupg", ".aws", ".azure",
        "id_rsa", "id_ed25519", ".env",
    )

    FORBIDDEN_FILENAMES = (
        "ntuser.dat", "ntuser.ini", "boot.ini", "bcd",
        "sam", "security", "software", "system",
        "pagefile.sys", "hiberfil.sys", "swapfile.sys",
        "hosts", "passwd", "shadow", "sudoers",
        "id_rsa", "id_ed25519", "authorized_keys",
        ".env", ".htaccess", ".gitconfig",
    )

    DANGEROUS_PHRASES = (
        "удали файл", "удали папку", "удали всё", "удали все",
        "выполни команду", "выполни код",
        "создай файл", "запиши файл", "перемести файл",
        "скопируй файл", "переименуй файл",
        "rm -rf", "rmdir /s", "rd /s", "del /f", "del /q",
        "format", "fdisk", "diskpart", "mkfs",
        "shutdown", "restart", "logoff",
        "reg delete", "reg add", "sc delete", "sc stop",
        "net user", "net localgroup", "net share",
        "runas", "takeown", "icacls", "cacls",
        "bcdedit", "bootrec", "sfc", "dism",
        "taskkill", "wmic", "powershell -enc",
        "curl | bash", "wget | sh", "iex(", "invoke-expression",
        "> con", "nul", "dd if=", "dd of=",
        "chmod 777", "chown root", "chattr",
        "sudo", "su -", "passwd", "useradd", "userdel",
    )

    BLOCKED_COMMAND_MARKERS = (
        "rm -rf", "rm -r", "rmdir /s", "rd /s",
        "del /f", "del /q", "del /s",
        "format ", "fdisk", "diskpart", "mkfs",
        "shutdown", "restart", "logoff",
        "reg delete", "reg add", "sc delete", "sc stop", "sc config",
        "net user", "net localgroup", "net share",
        "runas", "takeown", "icacls", "cacls",
        "bcdedit", "bootrec", "sfc ", "dism ",
        "taskkill", "wmic ", "powershell -enc", "powershell -e ",
        "curl | bash", "curl | sh", "wget | bash", "wget | sh",
        "iex(", "invoke-expression", "invoke-webrequest | iex",
        "> con", "nul", "dd if=", "dd of=",
        "chmod 777", "chown root", "chattr",
        "sudo ", "su -", "passwd", "useradd", "userdel",
        "& del", "& format", "&& del", "&& format",
        "| del", "| format", "for /f", "forfiles",
        "wevtutil cl", "cipher /w", "vssadmin delete",
        "wbadmin delete", "bcdedit /delete",
    )

    BLOCKED_CODE_MARKERS = (
        "shutil.rmtree", "os.remove", "os.rmdir", "os.unlink",
        "os.system", "os.popen", "os.exec", "os.spawn",
        "subprocess", "pty.spawn", "commands.getoutput",
        "eval(", "exec(", "compile(", "__import__(",
        "pickle.loads", "marshal.loads", "shelve.open",
        "open('/etc", 'open("C:\\Windows',
        'open("/etc', "open('C:\\Windows",
        "requests.", "urllib", "httpx.", "aiohttp.",
        "socket.socket", "telnetlib", "ftplib", "smtplib",
        "sys.exit", "os._exit", "quit(", "exit(",
        "globals()", "locals()", "vars(", "getattr(",
        "setattr(", "delattr(", "__builtins__", "__subclasses__",
        "ctypes", "cffi", "mmap", "resource.setrlimit",
        "signal.alarm", "threading.Thread", "multiprocessing",
        "tempfile.mktemp", "pathlib.Path.home",
        "Path.home", "expanduser", "os.environ[",
    )

    PROTECTED_ENV_VARS = (
        "PATH", "PYTHONPATH", "SYSTEMROOT", "WINDIR",
        "PROGRAMFILES", "PROGRAMDATA", "USERPROFILE",
        "APPDATA", "LOCALAPPDATA", "TEMP", "TMP",
    )

    ALLOWED_WRITE_DIRS = (
        "data", "models", "logs", "life", "cache", "tmp",
        "backups", "sandbox", "quarantine", "web", "tests",
    )

    ALLOWED_READ_DIRS = (
        "data", "models", "logs", "life", "cache", "tmp",
        "backups", "sandbox", "quarantine", "web", "tests",
    )

    ALLOWED_DELETE_DIRS = (
        "cache", "tmp", "sandbox", "quarantine",
    )

    ALLOWED_COMMAND_PREFIXES = (
        "dir", "ls", "type", "cat", "echo", "where", "which",
        "python --version", "python -c \"print",
        "pip list", "pip show",
    )

    LIFE_WEB_PAUSE_SEC = 25
    LIFE_STEP_PAUSE_SEC = 5
    LIFE_SLEEP_EVERY_STEPS = 1000
    LIFE_SLEEP_SEC = 60
    LIFE_MAX_STEPS_PER_SESSION = 500
    LIFE_AUTOLEARN_EVERY = 0  # Enable only after reviewing validation and verified examples
    LIFE_AUTOLEARN_EPOCHS = 1
    LIFE_MAX_WEB_SEARCHES_PER_STEP = 1
    LIFE_MAX_PAGE_CHARS = 30000
    LIFE_MAX_GOALS = 100000
    LIFE_JOURNAL_KEEP = 20000
    LIFE_JOURNAL_MAX_ENTRY_CHARS = 5000
    LIFE_CURIOSITY_ENABLED = True
    LIFE_CURIOSITY_MAX_NEW_GOALS = 2
    LIFE_ERRORS_TO_GOALS = False
    LIFE_SKIP_SERVICE_WORDS = True
    LIFE_MIN_TOPIC_LEN = 4
    LIFE_BOOTSTRAP_MIN_GOALS = 3
    LIFE_MIN_GOOD_PAIRS = 4
    LIFE_MAX_ANSWER_FOR_PAIR = 1500
    LIFE_MIN_ANSWER_FOR_PAIR = 60
    LIFE_MAX_TRAIN_TEXTS = 2000
    LIFE_SAFE_MODE = True
    LIFE_ALLOW_WRITES = True
    LIFE_ALLOW_DELETES = False
    LIFE_ALLOW_COMMANDS = False
    LIFE_ALLOW_CODE = False
    LIFE_ALLOW_NETWORK = True

    API_HOST = "127.0.0.1"
    API_PORT = 8765
    API_MAX_BODY_BYTES = 500_000
    API_CORS = True
    API_ENABLE_WEB_UI = True
    API_REQUIRE_TOKEN = False
    API_TOKEN = ""
    API_ALLOWED_HOSTS = ("127.0.0.1", "localhost")

    PLANNER_MAX_STEPS = 8
    PLANNER_SYSTEM_PROMPT_VERSION = "v2"

    RANDOM_SEED = 42
    VERBOSE = True
    DEBUG = False
    PROFILE = False
    LANGUAGE = "ru"
    TIMEZONE = "local"
    DATE_FORMAT = "%d.%m.%Y"
    TIME_FORMAT = "%H:%M:%S"
    DATETIME_FORMAT = "%d.%m.%Y %H:%M:%S"

    @classmethod
    def _all_dirs(cls):
        return [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.LIFE_DIR,
            cls.CACHE_DIR,
            cls.TMP_DIR,
            cls.WEB_DIR,
            cls.TESTS_DIR,
            cls.BACKUP_DIR,
            cls.SANDBOX_DIR,
            cls.QUARANTINE_DIR,
        ]

    @classmethod
    def init(cls):
        for directory in cls._all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        if cls.DEVICE == "cuda":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cls.CUDA_ALLOC_CONF)
        cls._log_security("Config initialized")

    @classmethod
    def _log_security(cls, message: str, level: str = "INFO"):
        try:
            cls.SECURITY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime(cls.DATETIME_FORMAT)
            with open(cls.SECURITY_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{stamp} [{level}] {message}\n")
        except OSError:
            pass

    @classmethod
    def is_dangerous(cls, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        return any(phrase in low for phrase in cls.DANGEROUS_PHRASES)

    @classmethod
    def normalize_path(cls, path) -> Path:
        try:
            return Path(path).expanduser().resolve(strict=False)
        except (OSError, ValueError, RuntimeError):
            return None

    @classmethod
    def is_forbidden_path(cls, path) -> bool:
        resolved = cls.normalize_path(path)
        if resolved is None:
            return True
        low = str(resolved).lower().replace("/", "\\")

        for forbidden in cls.FORBIDDEN_PATHS:
            if low.startswith(forbidden.lower().replace("/", "\\")):
                return True
        for part in cls.FORBIDDEN_PATH_PARTS:
            if part.lower() in low:
                return True
        if resolved.name.lower() in {n.lower() for n in cls.FORBIDDEN_FILENAMES}:
            return True
        return False

    @classmethod
    def is_within_base(cls, path) -> bool:
        resolved = cls.normalize_path(path)
        if resolved is None:
            return False
        base = cls.BASE_DIR.resolve()
        try:
            resolved.relative_to(base)
            return True
        except ValueError:
            return False

    @classmethod
    def _is_in_allowed_dir(cls, path, allowed_dirs) -> bool:
        resolved = cls.normalize_path(path)
        if resolved is None:
            return False
        for name in allowed_dirs:
            target = (cls.BASE_DIR / name).resolve()
            try:
                resolved.relative_to(target)
                return True
            except ValueError:
                continue
        return False

    @classmethod
    def can_read(cls, path) -> bool:
        if cls.READ_ONLY_MODE and not cls.is_within_base(path):
            return False
        if cls.is_forbidden_path(path):
            return False
        if not cls.is_within_base(path):
            return cls.ALLOW_EXTERNAL_PATHS and not cls.SANDBOX_MODE
        return cls._is_in_allowed_dir(path, cls.ALLOWED_READ_DIRS) or cls.is_within_base(path)

    @classmethod
    def can_write(cls, path) -> bool:
        if not cls.ALLOW_WRITE_OUTSIDE_BASE and not cls.is_within_base(path):
            return False
        if cls.is_forbidden_path(path):
            return False
        ext = Path(str(path)).suffix.lower()
        if ext and ext in cls.FORBIDDEN_EXTENSIONS:
            return False
        if ext and cls.ALLOWED_EXTENSIONS and ext not in cls.ALLOWED_EXTENSIONS:
            return False
        return cls._is_in_allowed_dir(path, cls.ALLOWED_WRITE_DIRS)

    @classmethod
    def can_delete(cls, path) -> bool:
        if not cls.ALLOW_DELETE_ANY:
            return False
        if cls.is_forbidden_path(path):
            return False
        if not cls.is_within_base(path):
            return False
        return cls._is_in_allowed_dir(path, cls.ALLOWED_DELETE_DIRS)

    @classmethod
    def can_execute_command(cls, command: str) -> bool:
        if not cls.ALLOW_COMMAND_EXECUTION:
            return False
        if not command:
            return False
        low = command.lower().strip()
        for marker in cls.BLOCKED_COMMAND_MARKERS:
            if marker in low:
                return False
        if ";" in command or "&&" in command or "||" in command or "|" in command:
            return False
        if ">" in command or "<" in command:
            return False
        if "`" in command or "$(" in command:
            return False
        first = low.split()[0] if low.split() else ""
        if first and not any(low.startswith(prefix) for prefix in cls.ALLOWED_COMMAND_PREFIXES):
            return False
        return True

    @classmethod
    def can_execute_code(cls, code: str) -> bool:
        if not cls.ALLOW_CODE_EXECUTION:
            return False
        if not code:
            return False
        low = code.lower()
        for marker in cls.BLOCKED_CODE_MARKERS:
            if marker.lower() in low:
                return False
        if "import os" in low and ("system" in low or "popen" in low):
            return False
        if "__" in code and "import" in low:
            return False
        return True

    @classmethod
    def can_use_network(cls) -> bool:
        return cls.ALLOW_NETWORK

    @classmethod
    def is_path_allowed(cls, path) -> bool:
        return cls.can_write(path)

    @classmethod
    def estimate_model_size_mb(cls):
        params = (
            cls.VOCAB_SIZE * cls.D_MODEL
            + cls.MAX_LEN * cls.D_MODEL
            + cls.N_LAYERS * (
                4 * cls.D_MODEL * cls.D_MODEL
                + 2 * cls.D_MODEL * cls.D_FF
            )
            + cls.D_MODEL * cls.VOCAB_SIZE
        )
        return round(params * 4 / (1024 ** 2), 1)

    @classmethod
    def estimate_training_peak_mb(cls):
        params_mb = cls.estimate_model_size_mb()
        activations = (
            cls.TRAIN_BATCH_SIZE * cls.TRAIN_MAX_LEN
            * cls.N_LAYERS * cls.D_MODEL * 20
        )
        activations_mb = activations * 4 / (1024 ** 2)
        optim = params_mb * 3
        return round(params_mb + activations_mb + optim, 1)

    @classmethod
    def estimate_inference_peak_mb(cls):
        params_mb = cls.estimate_model_size_mb()
        activations = (
            1 * cls.MAX_LEN * cls.N_LAYERS * cls.D_MODEL * 20
        )
        activations_mb = activations * 4 / (1024 ** 2)
        return round(params_mb + activations_mb, 1)

    @classmethod
    def estimate_free_vram_after_training(cls):
        total = cls.DEVICE_INFO.get("gpu_memory_gb") or 6.0
        peak_gb = cls.estimate_training_peak_mb() / 1024
        return round(total - peak_gb, 2)

    @classmethod
    def estimate_free_ram_after_training(cls):
        total = cls.SYSTEM_INFO.get("total_ram_gb") or 8.0
        peak_gb = cls.estimate_training_peak_mb() / 1024
        return round(total - peak_gb, 2)

    @classmethod
    def snapshot(cls):
        out = {}
        for key in dir(cls):
            if key.startswith("_"):
                continue
            value = getattr(cls, key)
            if callable(value):
                continue
            if isinstance(value, Path):
                value = str(value)
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, set):
                value = list(value)
            try:
                out[key] = value
            except Exception:
                out[key] = repr(value)
        out["snapshot_time"] = datetime.now().isoformat()
        out["model_size_mb"] = cls.estimate_model_size_mb()
        out["training_peak_mb"] = cls.estimate_training_peak_mb()
        out["inference_peak_mb"] = cls.estimate_inference_peak_mb()
        out["free_ram_after_training_gb"] = cls.estimate_free_ram_after_training()
        out["free_vram_after_training_gb"] = cls.estimate_free_vram_after_training()
        return out

    @classmethod
    def save_snapshot(cls):
        import json
        try:
            cls.CONFIG_SNAPSHOT.write_text(
                json.dumps(cls.snapshot(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    @classmethod
    def summary(cls):
        lines = [
            f"Device: {cls.DEVICE}",
            f"CPU count: {cls.SYSTEM_INFO.get('cpu_count')}",
            f"Python: {cls.SYSTEM_INFO.get('python_version')}",
            f"Platform: {cls.SYSTEM_INFO.get('platform')}",
            f"Base dir: {cls.BASE_DIR}",
            f"Model: d_model={cls.D_MODEL}, n_layers={cls.N_LAYERS}, "
            f"n_heads={cls.N_HEADS}, vocab={cls.VOCAB_SIZE}",
            f"Model size: ~{cls.estimate_model_size_mb()} MB",
            f"Training peak: ~{cls.estimate_training_peak_mb()} MB",
            f"Inference peak: ~{cls.estimate_inference_peak_mb()} MB",
            f"Free VRAM after training: ~{cls.estimate_free_vram_after_training()} GB",
            f"Free RAM after training: ~{cls.estimate_free_ram_after_training()} GB",
            f"VRAM limit: {cls.MAX_VRAM_GB} GB (reserved {cls.RESERVED_VRAM_GB} GB)",
            f"RAM limit: {cls.MAX_RAM_GB} GB (reserved {cls.RESERVED_RAM_GB} GB)",
            f"AMP: {cls.USE_AMP} ({cls.AMP_DTYPE})",
            f"Safe mode: {cls.SAFE_MODE}",
            f"Sandbox: {cls.SANDBOX_MODE}",
            f"Read-only: {cls.READ_ONLY_MODE}",
            f"Allow commands: {cls.ALLOW_COMMAND_EXECUTION}",
            f"Allow code: {cls.ALLOW_CODE_EXECUTION}",
            f"Allow deletes: {cls.ALLOW_DELETE_ANY}",
            f"Allow network: {cls.ALLOW_NETWORK}",
        ]
        if cls.SYSTEM_INFO.get("total_ram_gb"):
            lines.append(
                f"Total RAM: {cls.SYSTEM_INFO['total_ram_gb']} GB "
                f"(available: {cls.SYSTEM_INFO.get('available_ram_gb')} GB)"
            )
        if cls.DEVICE_INFO.get("gpu_name"):
            lines.append(
                f"GPU: {cls.DEVICE_INFO['gpu_name']} "
                f"({cls.DEVICE_INFO.get('gpu_memory_gb')} GB)"
            )
        return "\n".join(lines)

    @classmethod
    def validate(cls):
        errors = []
        if cls.D_MODEL % cls.N_HEADS != 0:
            errors.append("D_MODEL must be divisible by N_HEADS")
        if cls.TRAIN_BATCH_SIZE < 1:
            errors.append("TRAIN_BATCH_SIZE must be >= 1")
        if cls.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be > 0")
        if cls.MAX_LEN < cls.TRAIN_MAX_LEN:
            errors.append("MAX_LEN must be >= TRAIN_MAX_LEN")
        if cls.VOCAB_SIZE < len(cls.TOKENIZER_SPECIAL_TOKENS):
            errors.append("VOCAB_SIZE too small for special tokens")
        if cls.LIFE_MAX_STEPS_PER_SESSION < 1:
            errors.append("LIFE_MAX_STEPS_PER_SESSION must be >= 1")
        if cls.LIFE_AUTOLEARN_EVERY < 0:
            errors.append("LIFE_AUTOLEARN_EVERY must be >= 0")

        peak = cls.estimate_training_peak_mb()

        if cls.DEVICE == "cuda":
            vram_total = (cls.DEVICE_INFO.get("gpu_memory_gb") or 6.0) * 1024
            vram_limit = vram_total - cls.RESERVED_VRAM_GB * 1024
            if peak > vram_limit:
                errors.append(
                    f"Training peak ~{peak} MB exceeds available VRAM after "
                    f"reserve ({vram_limit:.0f} MB). Уменьши TRAIN_BATCH_SIZE, "
                    f"D_MODEL или N_LAYERS."
                )

        ram_total = (cls.SYSTEM_INFO.get("total_ram_gb") or 8.0) * 1024
        ram_limit = ram_total - cls.RESERVED_RAM_GB * 1024
        if peak > ram_limit:
            errors.append(
                f"Training peak ~{peak} MB exceeds available RAM after "
                f"reserve ({ram_limit:.0f} MB)."
            )
        return errors

    @classmethod
    def memory_report(cls):
        lines = []
        try:
            import psutil
            vm = psutil.virtual_memory()
            lines.append(
                f"RAM total: {_bytes_to_gb(vm.total):.2f} GB, "
                f"available: {_bytes_to_gb(vm.available):.2f} GB, "
                f"used: {vm.percent}%, "
                f"process RSS: {_bytes_to_mb(psutil.Process().memory_info().rss):.0f} MB"
            )
        except Exception:
            lines.append("psutil not available")
        if cls.DEVICE == "cuda" and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                free, total = torch.cuda.mem_get_info()
                lines.append(
                    f"VRAM allocated: {allocated:.0f} MB, "
                    f"reserved: {reserved:.0f} MB, "
                    f"free: {_bytes_to_gb(free):.2f} GB / "
                    f"{_bytes_to_gb(total):.2f} GB"
                )
            except Exception as error:
                lines.append(f"CUDA memory info error: {error}")
        return "\n".join(lines)

    @classmethod
    def security_report(cls):
        lines = [
            "=== SECURITY ===",
            f"SAFE_MODE: {cls.SAFE_MODE}",
            f"SANDBOX_MODE: {cls.SANDBOX_MODE}",
            f"READ_ONLY_MODE: {cls.READ_ONLY_MODE}",
            f"CONFIRM_DANGEROUS: {cls.CONFIRM_DANGEROUS}",
            f"REQUIRE_CONFIRMATION_CODE: {cls.REQUIRE_CONFIRMATION_CODE}",
            f"ALLOW_COMMAND_EXECUTION: {cls.ALLOW_COMMAND_EXECUTION}",
            f"ALLOW_CODE_EXECUTION: {cls.ALLOW_CODE_EXECUTION}",
            f"ALLOW_DELETE_ANY: {cls.ALLOW_DELETE_ANY}",
            f"ALLOW_NETWORK: {cls.ALLOW_NETWORK}",
            f"ALLOW_WRITE_OUTSIDE_BASE: {cls.ALLOW_WRITE_OUTSIDE_BASE}",
            f"ALLOW_EXTERNAL_PATHS: {cls.ALLOW_EXTERNAL_PATHS}",
            f"MAX_OPERATIONS_PER_MINUTE: {cls.MAX_OPERATIONS_PER_MINUTE}",
            f"MAX_DELETES_PER_SESSION: {cls.MAX_DELETES_PER_SESSION}",
            f"MAX_WRITES_PER_SESSION: {cls.MAX_WRITES_PER_SESSION}",
            f"MAX_COMMANDS_PER_SESSION: {cls.MAX_COMMANDS_PER_SESSION}",
            f"MAX_CODE_RUNS_PER_SESSION: {cls.MAX_CODE_RUNS_PER_SESSION}",
        ]
        return "\n".join(lines)