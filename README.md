# KoKs

Обновлённый порядок обучения и ограничения: [BALANCED_LEARNING.md](BALANCED_LEARNING.md).
Автообучение весов и выполнение Python отключены по умолчанию.

Локальный русскоязычный ассистент с памятью, базой знаний, инструментами компьютера, собственной Transformer-моделью и автономным циклом жизни. Оптимизирован под 8 ГБ RAM и 6 ГБ VRAM (RTX 3050 Laptop). Включает полноценную многоуровневую защиту.

## Требования

- Python 3.10+
- PyTorch с CUDA (для GPU) или CPU-сборка
- 8 ГБ RAM, 6 ГБ VRAM (для RTX 3050)
- Опционально: psutil

## Установка

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install psutil tqdm numpy

## Запуск

python main.py --load
python main.py --load --query "который час"
python main.py --train data/dialogs.txt --epochs 10

## Безопасность

Ассистент работает в закрытом режиме. Все опасные операции контролируются через `tools/security.py` и `Config`.

### Режимы по умолчанию

- SAFE_MODE = True
- SANDBOX_MODE = True
- READ_ONLY_MODE = False
- ALLOW_COMMAND_EXECUTION = False
- ALLOW_DELETE_ANY = False
- ALLOW_WRITE_OUTSIDE_BASE = False
- ALLOW_EXTERNAL_PATHS = False
- ALLOW_NETWORK = True
- ALLOW_CODE_EXECUTION = False

### Что запрещено полностью

- `execute_command` — отключено.
- `delete_file` — отключено.
- `удали файл`, `удали папку` — отключено.
- `открой файл`, `открой папку` через ОС — отключено.
- `выполни команду` — отключено.
- Запись и чтение вне папки проекта — запрещено.
- Пути `C:\Windows`, `C:\Program Files`, `C:\ProgramData`, `/etc`, `/usr`, `/bin` — запрещены.
- Части пути `system32`, `syswow64`, `appdata`, `recycle.bin` — запрещены.
- Имена файлов `ntuser.dat`, `sam`, `security`, `hosts`, `id_rsa` — запрещены.
- Расширения `.exe`, `.bat`, `.cmd`, `.ps1`, `.dll`, `.reg` — запрещены.

### Уровни защиты

**Config.can_read / can_write / can_delete**

Проверяют путь, расширение, наличие запрещённых частей, принадлежность проекту, наличие в белом списке.

**Config.can_execute_command**

Белый список префиксов (`dir`, `ls`, `cat`, `echo`), блокировка маркеров (`rm -rf`, `format`, `reg delete`, `shutdown`, `takeown`, `icacls`, `powershell -enc`, `iex(`, `sudo`, `passwd`, `useradd`). Также запрещены `;`, `&&`, `||`, `|`, `>`, `<`, `` ` ``, `$(`.

**Config.can_execute_code**

Блокировка `shutil.rmtree`, `os.remove`, `os.system`, `subprocess`, `eval`, `exec`, `compile`, `__import__`, `pickle.loads`, `socket.socket`, `ctypes`, `Path.home`, `expanduser`, `sys.exit`. Код запускается в `sandbox/`.

**SecurityManager**

- Счётчики за сессию: записи, удаления, команды, код.
- Ограничение частоты: `MAX_OPERATIONS_PER_MINUTE`.
- Блокировка после подозрительных попыток: `SUSPICIOUS_ACTIONS_TO_LOCK`.
- Персистентность в `security_state.json`.
- Подтверждение опасных действий кодом: `REQUIRE_CONFIRMATION_CODE`.

### Логи безопасности

Все отказы пишутся в `logs/security.log` с уровнем `WARN` или `ERROR`.

### Команды чата

/security    - отчёт безопасности
/lock        - заблокировать все операции
/unlock      - разблокировать

### Проверка

python main.py --security-report

Выводит текущие флаги `Config` и счётчики `SecurityManager`.

### Важно

- Не запускай ассистента от имени администратора.
- Не устанавливай `ALLOW_COMMAND_EXECUTION = True` без изоляции.
- Не устанавливай `ALLOW_DELETE_ANY = True`.
- Не отключай `SANDBOX_MODE`.
- При подозрительной активности `SecurityManager` сам блокирует работу.

## Понимание вопросов

Ассистент понимает вопросы семантически.

- Стемминг русских и английских слов.
- Синонимы: «ИИ» = «искусственный интеллект» = «AI», «Python» = «питон».
- Тип вопроса: «что такое X» — определение, «как X» — инструкция, «почему X» — причина.
- TF-IDF-оценка: редкие слова важнее частых.
- Очистка ответа: навигация Википедии, шапки сайтов отсеиваются.

Поиск по трём источникам:

1. Точное совпадение (dialogs.txt, teacher_dialogs.txt).
2. Семантический поиск по диалогам.
3. Семантический поиск по знаниям.

## База знаний

python main.py --load --import-knowledge notes.txt README.md

В интерактивном режиме:

запомни файл notes.txt

## HTTP API

python main.py --load --api --host 127.0.0.1 --port 8765

Маршруты: GET /health, GET /memory, POST /query.
По умолчанию слушает только 127.0.0.1.

## Обучение

python main.py --train data/dialogs.txt --epochs 10

Модель: d_model=320, n_layers=6, n_heads=8, d_ff=1280, vocab=24000, max_len=512.
Под RTX 3050 6 ГБ: TRAIN_BATCH_SIZE=16, TRAIN_MAX_LEN=192, AMP в bf16.

Оценки:

from config import Config
print(Config.estimate_model_size_mb())
print(Config.estimate_training_peak_mb())
print(Config.estimate_inference_peak_mb())
print(Config.estimate_free_vram_after_training())
print(Config.validate())
print(Config.summary())
print(Config.memory_report())
print(Config.security_report())

### Обучение через живой диалог

/teach       включить запись вопросов и ответов
/stop-teach  остановить запись
/train-now   дообучить модель

Для исправления ответа: «правильно <верный ответ>».

## Автономная жизнь

python main.py --life
python main.py --life --life-steps 100

Агент ставит цели, ищет информацию, читает страницы и сохраняет кандидатов на проверку. Автоматическое изменение весов по умолчанию отключено.

Журнал: life/state/journal.jsonl. Цели: life/state/goals.json.

### Что разрешено в режиме жизни

- читать интернет (поиск + чтение страниц);
- писать только в свои папки: life/state/*, data/*, knowledge.json, memory.json, models/*;
- дообучать модель.

### Что запрещено

- execute_command, delete_file, write_file вне проекта;
- удаление любых файлов;
- выполнение команд.

### Лимиты

- LIFE_STEP_PAUSE_SEC
- LIFE_SLEEP_EVERY_STEPS, LIFE_SLEEP_SEC
- LIFE_MAX_STEPS_PER_SESSION
- LIFE_AUTOLEARN_EVERY
- LIFE_MAX_GOALS, LIFE_JOURNAL_KEEP
- LIFE_MAX_TRAIN_TEXTS

### Фильтрация обучающих пар

Перед сохранением веб-кандидата в learning_records.json (verified=false):

- вопрос не является служебным словом;
- вопрос не содержит маркеров ошибок;
- ответ не является навигацией сайта;
- ответ содержит связный текст;
- пары с одинаковым вопросом не дублируются.

### Извлечение ответа со страницы

Из страницы берётся фрагмент, релевантный вопросу:

- страница разбивается на предложения;
- ключевые слова вопроса ищутся в предложениях;
- выбирается лучшее предложение и следующее за ним;
- результат обрезается до 600 символов.

## Совместимость чекпоинтов

Размеры модели (`vocab_size`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `max_len`) читаются из чекпоинта через `read_checkpoint_shape` в `core/trainer.py`.

## Структура проекта

agent/
  dialogs.py        семантический поиск по dialogs.txt и teacher_dialogs.txt
  knowledge.py      база знаний с семантическим поиском
  memory.py         краткосрочная память
  executor.py       реестр инструментов
  planner.py        построение system prompt
  semantics.py      стемминг, синонимы, типы вопросов, TF-IDF, фильтры ответов
core/
  model.py          Transformer-модель
  tokenizer.py      BPE-токенизатор
  trainer.py        обучение, сохранение, чтение размеров чекпоинта
  inference.py      генерация
life/
  life_loop.py      автономный цикл жизни
  journal.py        журнал жизни
  goals.py          цели агента
  curiosity.py      самопостановка целей
  autolearn.py      самообучение и извлечение ответов со страниц
tools/
  security.py       централизованный контроль опасных операций
  file_tools.py     файловые операции с проверкой пути
  system_tools.py   системные команды с блокировкой
  code_tools.py     выполнение Python в песочнице с блокировкой
  web_tools.py      поиск в интернете
  read_tools.py     чтение страниц с распаковкой gzip
api.py              HTTP API
config.py           конфигурация, лимиты, безопасность
main.py             точка входа

## Проверка

python -m unittest discover -s tests -v#   K . O . K . S  
 