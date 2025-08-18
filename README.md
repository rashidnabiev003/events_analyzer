## events_analyzer

Инструмент для автоматизированной обработки описаний мероприятий из XLSX:

- **Обогащение метаданными** при помощи LLM (регион, ресурсы/подрядчики)
- **Семантический поиск** близких мероприятий (эмбеддинги + FAISS, опц. реранк)
- **Оценка риска связности** пар мероприятий при помощи LLM и формирование risk‑matrix

Подходит для пакетной обработки отдельных `.xlsx` файлов и целых папок.

### Кратко
- Вход: таблица XLSX с колонками `id`, `title`, `description` и последним столбцом `year`
- Выход: `raw.csv` → `enriched_<name>.csv` → `search_index/*` → `risk_<name>.csv`
- Движок LLM: удалённый OpenWebUI API или локальный vLLM

---

## Возможности
- **Импорт из XLSX**: чтение нужных столбцов, очистка и склейка текста
- **Метаданные из LLM**: для каждого события извлекаются `region` и `resources`
- **FAISS‑индекс**: построение индекса сходства по эмбеддингам BGE‑M3
- **Реранкинг (опционально)**: BGE‑M3 reranker на CPU/GPU для улучшения качества
- **Подбор кандидатов**: отбор пар похожих мероприятий с порогами/квотами
- **Оценка риска**: LLM формирует JSON `{"risk": float, "reason": str}` для каждой пары
- **CLI**: один исполняемый модуль, обработка файла или папки, батч‑режим

---

## Архитектура и ключевые модули
- `src/model.py` — сценарии обогащения и расчёта risk‑matrix, CLI
- `src/utils/data_loader.py` — преобразование XLSX → `raw.csv`
- `src/utils/embeddings_search.py` — эмбеддинги, FAISS‑индекс, подбор пар
- `src/openwebui_engine.py` — клиент OpenWebUI (OpenAI‑совместимый API)
- `src/vllm_engine.py` — локальный vLLM движок для батч‑инференса
- `src/schemas/main_schemas.py` — Pydantic‑схемы конфигурации и ответов

---

## Требования
- Python `3.12`
- Рекомендуется менеджер окружений `uv` или `venv`
- Для продакшн‑семантики: `faiss-cpu` или `faiss-gpu` и `sentence-transformers`
- Для реранка (опционально): `FlagEmbedding` (BGE‑M3)
- Для локального LLM: `vllm` и модель, указанная в конфиге

> В тестовых/ограниченных окружениях проект имеет фоллбеки: если нет FAISS или
> модель эмбеддингов не может быть загружена из интернета, используется детерминированный
> NumPy‑индекс и простая hashing‑эмбеддер. Это позволяет запускать тесты без сети/GPU.

---

## Установка

### Вариант 1: через `uv`
```bash
uv venv
uv pip install -e .[dev]
```

### Вариант 2: через `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

> PyTorch и FAISS могут требовать специфичных колёс под вашу платформу/GPU.
> При необходимости укажите индекс PyTorch (`pyproject.toml` уже содержит пример для cu128).

---

## Конфигурация

Основной конфиг локального vLLM: `src/configs/config.json`.

Пример (из репозитория):
```json
{
  "vllm_engine_config": {
    "model_path": "/home/rashid/models/Qwen3-8B/",
    "gpu_memory_utilization": 0.7,
    "trust_remote_code": true,
    "quantization": "fp8",
    "tensor_parallel_size": 2,
    "max_model_length": 8192,
    "max_batch_size": 1000
  }
}
```

### Переменные окружения
- `USE_OPENWEBUI` — `1` (по умолчанию) для использования OpenWebUI, `0` — использовать локальный `vllm`
- `OPENWEBUI_BASE_URL` — базовый URL OpenWebUI (например, `https://webui.example.com`)
- `OPENWEBUI_API_KEY` — токен доступа
- `OPENWEBUI_MODEL` — имя модели в OpenWebUI (например, `gpt-oss:120b`)
- `OPENWEBUI_TIMEOUT` — таймаут HTTP (сек)
- `OPENWEBUI_CONCURRENCY` — максимальное число одновременных запросов

---

## Форматы данных

### Вход: XLSX
Ожидается лист с данными, где:
- Колонка `0` — идентификатор (`event_id`)
- Колонки `1` и `2` — заголовок/описание, склеиваются в `raw_text`
- Последняя колонка листа — трактуется как `year`

По умолчанию берётся лист `"Лист2"`. Это можно переопределить при вызове `build_raw`.

### Промежуточные/выходные файлы
- `raw.csv` — результат этапа импорта
  - Колонки: `event_id`, `raw_text`, `year`
- `enriched_<name>.csv` — добавленные метаданные из LLM
  - Колонки: `event_id`, `raw_text`, `year`, `region`, `resources`
  - `resources` — строка, где список ресурсов соединён `;`
- `search_index/`
  - `events.faiss` — FAISS‑индекс (или плейсхолдер, если активирован фоллбек)
  - `embeddings.npy` — матрица эмбеддингов
  - `metadata.json` — список `{event_id, text}`
- `risk_<name>.csv` — оценённые пары событий
  - Колонки: `A_id`, `B_id`, `risk`, `reason`

---

## Быстрый старт

### Обработка одного XLSX файла (OpenWebUI)
```bash
USE_OPENWEBUI=1 \
OPENWEBUI_BASE_URL="https://webui.g-309.ru" \
OPENWEBUI_API_KEY="<token>" \
OPENWEBUI_MODEL="gpt-oss:120b" \
python -m src.model process /path/to/input.xlsx /path/to/output
```

### Обработка папки с XLSX (локальный vLLM)
```bash
USE_OPENWEBUI=0 \
python -m src.model process /path/to/folder_with_xlsx /path/to/output
```

В процессе:
1) XLSX → `raw.csv`
2) LLM обогащает → `enriched_<name>.csv` (батчами с инкрементальной записью)
3) Строится `search_index/*`
4) Подбираются пары похожих событий и для них считается риск → `risk_<name>.csv`

---

## Как это работает

### Метаданные (LLM)
Модель получает промпт с текстом события и должна вернуть строго JSON:
```json
{"region": "<строка>", "resources": ["<строка>", "..."]}
```
Поля валидируются по схеме `Clast`.

### Семантический поиск
`EventsSemanticSearch`:
- Эмбеддинги `sentence-transformers` (BGE‑M3), устройство: CPU или CUDA
- Нормализация и индекс `IndexFlatIP` (FAISS)
- Опциональный реранк через `FlagEmbedding` BGEM3 (`compute_score`)

Отбор пар: `make_pairs_percent(...)`:
- `k_preselect` — сколько соседей взять из FAISS на событие
- `min_faiss_sim` — отсечка слабых по FAISS до реранка
- `sim_threshold` — финальный порог «процента схожести» после реранка/FAISS
- `keep_top_pct` — альтернатива порогу: оставить верхний процент лучших
- `per_event_cap` — ограничение числа пар на событие
- `dedup_bidirectional` — избавление от дубликатов (A,B)==(B,A)

### Расчёт риска (LLM)
Для каждой пары (A,B) формируется промпт. Модель должна вернуть JSON вида:
```json
{"risk": 0.25, "reason": "...пояснение связности..."}
```
Где `risk ∈ [0,1]`. Внутренние инструкции подсказывают модели вычислять вероятность через
дискретные уровни влияния (время/ресурсы), но наружу возвращается только итог.

---

## CLI
```bash
python -m src.model process <input_path> <output_dir>
```
- **input_path** — путь к `.xlsx` файлу или папке с `.xlsx`
- **output_dir** — каталог для результатов

Пример вывода:
- `raw.csv`, `enriched_<name>.csv`, `search_index/*`, `risk_<name>.csv`

---

## Тесты
```bash
uv run pytest -q
```
Проект содержит фоллбеки для окружений без FAISS/интернета, поэтому тесты
должны проходить «из коробки».

---

## Производительность и рекомендации
- Для больших датасетов используйте GPU для эмбеддингов и реранка
- Настройте `max_batch_size` для LLM/векторизатора
- Для OpenWebUI увеличьте `OPENWEBUI_CONCURRENCY`, если сервер позволяет
- Для FAISS можно перенести индекс на GPU (в коде сейчас CPU‑вариант)

---

## Частые проблемы и решения
- **FAISS не установлен**: будет использоваться NumPy‑фоллбек. Для продакшна установите
  `faiss-cpu` или `faiss-gpu` под вашу платформу.
- **Модель эмбеддингов не скачивается**: включится hashing‑фоллбек. Установите
  `sentence-transformers` и проверьте доступ в интернет или скачайте веса заранее.
- **OpenWebUI 401/403**: проверьте `OPENWEBUI_API_KEY` и URL, а также права модели.
- **Недостаточно VRAM**: уменьшайте `max_batch_size`, включите квантование, снизьте
  `gpu_memory_utilization` или перераспределите `tensor_parallel_size`.
- **XLSX не читается**: укажите корректный `sheet_name` и индексы колонок.

---

## Структура репозитория (важное)
- `main.py` — простой корневой файл (не используется в основном сценарии)
- `src/` — исходники
  - `configs/config.json` — конфиг vLLM
  - `model.py` — CLI и оркестрация пайплайна
  - `utils/data_loader.py` — XLSX → CSV
  - `utils/embeddings_search.py` — эмбеддинги/FAISS/подбор пар
  - `openwebui_engine.py` — клиент OpenWebUI
  - `vllm_engine.py` — локальный LLM через vLLM
  - `schemas/main_schemas.py` — Pydantic‑схемы
  - `data/` — пример входных данных
  - `data_/` — пример выходных/рабочих файлов
- `tests/` — автотесты

---

## Лицензия
Укажите лицензию проекта при необходимости.
