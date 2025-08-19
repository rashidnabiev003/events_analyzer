import json
import re
import csv
import asyncio  
import sys
import time
import argparse
import os
from pathlib import Path
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))
try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass
from typing import Any, Dict, List
from src.utils.data_loader import build_raw
from src.schemas.main_schemas import (AppConfig,
                                        Clast,
                                        RiskResp,
                                        Description,
                                        ChatItem)
from typing import Optional, List
from pydantic import TypeAdapter
import pandas as pd
import requests
from tqdm import tqdm
from src.utils.embeddings_search import EventsSemanticSearch, BuildConfig

CONFIG_PATH = CONFIG_PATH = ROOT / "src" / "configs" / "config.json" 
def load_config(path: Path = CONFIG_PATH) -> AppConfig:
	txt = Path(path).read_text(encoding="utf-8")
	return TypeAdapter(AppConfig).validate_json(txt)


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.I)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def chat(*args, **kwargs) -> str:
	"""Lightweight stub for tests; real chatting is handled by VLLMEngine in production paths."""
	return "{}"


SYSTEM_PROMPT = (
   """
Ты — эксперт по анализу рисков в проектах национальных мероприятий. Перед тобой пара событий:
- **Исходное мероприятие** (то, влияние на которое нужно оценить).
- **Наиболее связное мероприятие** (то, которое столкнулось со сбоем: отмена, задержка или существенное изменение).

Предположим, что связное мероприятие столкнулось со сбоем. Тебе нужно, следуя строгой методологии, оценить влияние этого сбоя на исходное событие и вывести результат в виде JSON‑объекта. 

1. **Оцени влияние** по трём независимым критериям, присваивая каждому целое значение от 0 до 4, где 0 — отсутствует, 4 — критическое:
   - `time_level` — насколько сбой влияния на сроки и выполнение графика.
   - `resource_level` — насколько это влияет на ресурсы (финансовые, человеческие, материальные).
   - `scope_level` — насколько это влияет на объём/содержание мероприятия (масштаб, цели, требования).
   Выбирай оценку, опираясь на тексты мероприятий, но не выводи её в ответ.

2. **Скрыто** (про себя) вычисли:
   - `total_risk = time_level + resource_level + scope_level` (диапазон 0–12).
   - `probability_failure = total_risk / 12.0`.
   Эти вычисления служат только для определения итогового риска; **не выводи сами расчёты и уровни в поле `reason`**.

3. **Сформируй JSON‑ответ** с двумя полями:
   - `"risk"` — это значение `probability_failure`, записанное десятичной точкой и двумя знаками после неё.
   - `"reason"` — краткое текстовое объяснение влияния, где:
     * укажи коротко, что представляют собой исходное и связное мероприятия,
     * поясни словами, почему сбой связного влияет (или не влияет) на исходное в рамках трёх критериев,
     * **не включай в reason никакие числа, формулы, расчёты или упоминания оценок**,
     * соблюдай правило: мероприятие года X может зависеть от мероприятий того же года или предыдущих лет, но не от будущих (например, событие 2025 года может зависеть от 2024 и 2025, но не от 2026; событие 2026 может зависеть от 2025 и 2026).

4. **Формат вывода**: выведи строго один JSON‑объект без Markdown и без пояснений. Пример структуры:

{
  "risk": <float>  # ДОЛЖЕН равняться probability_failure
  "reason": "Исходное мероприятие: ...; Связное мероприятие: ...; Обоснование зависимости словами без расчётов."
}
"""
)
SYSTEM_METADATA_PROMPT = (
"""
Ты извлекаешь метаданные. Верни СТРОГО один валидный JSON-объект ровно с ключами:
{"region": "<строка>", "resources": ["<строка>", ...]}

Только JSON, без какого-либо текста/Markdown до или после.
Если данных нет — region="", resources=[].
Ответ ДОЛЖЕН начинаться с "{" и заканчиваться "}".
"""
)

METADATA_PROMPT = """
Извлеки JSON с полями:
region  – регион (строка),
resources – список ключевых ресурсов или подрядчиков (массив строк).
{{
  "region": <str>,
  "resources": <массив str>
}}
Текст мероприятия:
\"\"\"{text}\"\"\"
"""

RISK_PROMPT = """
Вот 2 мероприятия их описания и названия. Тебе нужно объяснить связность этих мероприятий. Рассуждай логически, точно и цельно.
- Исходное мероприятие : {a_text}  
- Связное мероприятие : {b_text}

json
**Формат ответа (строго JSON):**
{{
  "risk": probability_failure,
  "reason": "Исходное мероприятие: ...\nНаиболее связанное мероприятие: ...\nОбъяснение зависимости: ..."
}}
Начинайте сразу после этого сообщения.
"""

cfg = load_config()        
RAW_DIR = Path("data")
RAW_DIR.mkdir(exist_ok=True)

# Глобальный движок (для тестов может быть подменён monkeypatch'ем)
engine = None


def enrich_with_metadata_df(
    df: pd.DataFrame,
    vllm_engine,
    json_schema: Any,
    out_csv: Path | None = None,
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Обогащает df метаданными через пакетные запросы vllm/webui.
    Пишет результат в CSV по частям (батчами), если задан out_csv.
    Возвращает также полный DataFrame (собираем в памяти).
    """
    # гарантируем колонки
    for col in ("region", "resources"):
        if col not in df.columns:
            df[col] = ""

    items_all: list[ChatItem] = [
        ChatItem(system_prompt=SYSTEM_METADATA_PROMPT,
                 prompt=METADATA_PROMPT.format(text=raw_text))
        for raw_text in df["raw_text"].astype(str).tolist()
    ]

    header_cols = ["event_id"]
    if "np_name" in df.columns:
        header_cols.append("np_name")
    if "year" in df.columns:
        header_cols.append("year")
    header_cols += ["raw_text", "region", "resources"]

    # шапка
    df.head(0)[header_cols].to_csv(out_csv, index=False, encoding="utf-8")

    from tqdm import tqdm
    for start in tqdm(range(0, len(items_all), batch_size), desc="enrich", unit="row"):
        end = start + batch_size
        batch_items = items_all[start:end]
        responses = vllm_engine.chat_batch(items=batch_items, json_schema=json_schema)

        # переносим ответы в df
        for k, resp in enumerate(responses, start=start):
            df.at[k, "region"] = resp.get("region", "")
            res = resp.get("resources", [])
            if isinstance(res, list):
                res = ";".join(res)
            df.at[k, "resources"] = res

        # инкрементально дозаписываем только обновлённый срез
        df_slice = df.iloc[start:end][header_cols]
        df_slice.to_csv(out_csv, index=False, header=False, mode="a", encoding="utf-8")

    return df


def risk_matrix(
    enriched_csv: Path = RAW_DIR / "enriched.csv",
    first_col_l: int = 0,
    first_col_u: int = 10,
    second_col_l: int = 0,
    second_col_u: int = 50,
    out_csv: Path = RAW_DIR / "risk_matrix.csv",
    candidate_pairs: Optional[List[tuple[int, int]]] = None,
    engine=None,
) -> None:
    df = pd.read_csv(enriched_csv)

    if not candidate_pairs:
        targets = df.iloc[first_col_l:first_col_u]
        candidates = df.iloc[second_col_l:second_col_u]
        candidate_pairs = [
            (str(a.event_id), str(b.event_id))
            for _, a in targets.iterrows()
            for _, b in candidates.iterrows()
            if str(a.event_id) != str(b.event_id)
        ]

    id_to_row = {int(r.event_id): r for _, r in df.iterrows()}
    all_prompts: List[ChatItem] = []
    all_pairs: List[tuple[int, int]] = []

    for a_id_str, b_id_str in tqdm(candidate_pairs, total=len(candidate_pairs), desc="Collecting pairs"):
        a_id, b_id = int(a_id_str), int(b_id_str)
        row_a, row_b = id_to_row[a_id], id_to_row[b_id]

        a_text = (
            f"Текст: {row_a.raw_text}\n"
            f"Регион: {row_a.region}; Год: {getattr(row_a, 'year', '')}; "
            f"Ресурсы: {row_a.resources}"
        )
        b_text = (
            f"Текст: {row_b.raw_text}\n"
            f"Регион: {row_b.region}; Год: {getattr(row_b, 'year', '')}; "
            f"Ресурсы: {row_b.resources}"
        )

        prompt = RISK_PROMPT.format(a_text=a_text, b_text=b_text)
        all_prompts.append(ChatItem(prompt=prompt, system_prompt=SYSTEM_PROMPT))
        all_pairs.append((a_id, b_id))

    total = len(all_prompts)
    bs = cfg.vllm_engine_config.max_batch_size

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile, \
         tqdm(total=total, desc="risk", unit="pair") as pbar:
        writer = csv.DictWriter(csvfile, fieldnames=["A_id", "B_id", "risk", "reason"])
        writer.writeheader()

        for i in range(0, total, bs):
            batch_prompts = all_prompts[i:i + bs]
            batch_pairs   = all_pairs[i:i + bs]

            batch_responses = engine.chat_batch(batch_prompts, json_schema=RiskResp) if engine else [
                {"risk": 0.0, "reason": ""} for _ in batch_prompts
            ]

            for (a_id, b_id), response in zip(batch_pairs, batch_responses):
                writer.writerow({
                    "A_id": a_id,
                    "B_id": b_id,
                    "risk": float(response.get("risk", 0.0)),
                    "reason": response.get("reason", "")
                })

            csvfile.flush()
            pbar.update(len(batch_prompts))

    print(f"✔ risk matrix → {out_csv}")



def get_xlsx_files(input_dir: Path) -> list[Path]:
    """
    Возвращает список всех .xlsx файлов в папке input_dir.
    """
    return sorted(input_dir.glob("*.xlsx"))


def process_file(
    xlsx_path: Path,
    output_dir: Path,
    engine,
) -> None:
    name = xlsx_path.stem

    raw_csv = build_raw(
        xlsx_path=xlsx_path,
        column_idx=(0, 1, 2),
        out_path=output_dir / "raw.csv",
    )
    df = pd.read_csv(raw_csv)

    enriched_csv = output_dir / f"enriched_{name}.csv"
    enriched_df = enrich_with_metadata_df(
    df=df,
    vllm_engine=engine,
    json_schema=Clast,
    out_csv=enriched_csv,          # <<< инкрементальная запись
    batch_size=128
)
    print(f"✔ enriched CSV → {enriched_csv}")

    search_dir = output_dir / "search_index"
    searcher = EventsSemanticSearch(
        workdir=search_dir,
        cfg=BuildConfig(
            force_cpu=True,          # эмбеддер/FAISS на CPU
            enable_rerank=True,
            rerank_device="cpu",     # или "cuda" при наличии VRAM
            rerank_batch_size=2048,
            use_fp16_rerank=True
        )
    )
    searcher.build_from_dataframe(enriched_df, text_col="raw_text", id_col="event_id")
    pairs = searcher.make_pairs_percent(
        k_preselect=50,
        min_faiss_sim=0.20,
        sim_threshold=0.50,      # «процент схожести» как порог
        keep_top_pct=0.30,
        per_event_cap=50,
        id_col="event_id",
        dedup_bidirectional=True,
        use_reranker=True
    )    
    if "np_name" in enriched_df.columns:
        id_to_np = {str(r.event_id): str(r.np_name) for _, r in enriched_df.iterrows()}
        pairs = [(a, b) for (a, b) in pairs if id_to_np.get(str(a)) == id_to_np.get(str(b))]

    # Risk matrix
    risk_csv = output_dir / f"risk_{name}.csv"
    risk_matrix(
        enriched_csv=enriched_csv,
        first_col_l=0, first_col_u=len(enriched_df),
        second_col_l=0, second_col_u=len(enriched_df),
        out_csv=risk_csv,
        candidate_pairs=pairs,
        engine=engine
    )
    print(f"✔ processed {name} → {risk_csv}")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    engine,
) -> None:
    """
    Обходит все xlsx-файлы в input_dir и обрабатывает каждый через process_file().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for xlsx_path in get_xlsx_files(input_dir):
        process_file(
            xlsx_path=xlsx_path,
            output_dir=output_dir,
            engine=engine
        )

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="events_analyzer",
        description="Обработка XLSX: обогащение и расчет risk matrix"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "process", help="Обработать файл или папку с XLSX"
    )
    sp.add_argument(
        "input_path", type=Path,
        help=".xlsx файл или папка с .xlsx"
    )
    sp.add_argument(
        "output_dir", type=Path,
        help="Куда сохранять результаты"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    USE_OPENWEBUI = os.getenv("USE_OPENWEBUI", "1") == "1"
    print(USE_OPENWEBUI)
    # Инициализация движка VLLM 
    if USE_OPENWEBUI:
        from src.openwebui_engine import OpenWebUIEngine
        engine = OpenWebUIEngine(
            base_url=os.getenv("OPENWEBUI_BASE_URL", "https://webui.g-309.ru"),
            api_key=os.getenv("OPENWEBUI_API_KEY", ""),
            model=os.getenv("OPENWEBUI_MODEL", "gpt-oss:120b"),
            timeout=float(os.getenv("OPENWEBUI_TIMEOUT", "60")),
            max_concurrency=int(os.getenv("OPENWEBUI_CONCURRENCY", "16")),
            system_prompt=SYSTEM_PROMPT,
        )
    else:
        from src.vllm_engine import VLLMEngine
        cfg = load_config()  # убедимся, что cfg загружен здесь
        engine = VLLMEngine(engine_config=cfg.vllm_engine_config, system_prompt=SYSTEM_PROMPT)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_path.is_dir():
        print(f"[process] folder: {args.input_path} -> {args.output_dir}")
        process_folder(args.input_path, args.output_dir, engine)
    else:
        print(f"[process] file:   {args.input_path} -> {args.output_dir}")
        process_file(args.input_path, args.output_dir, engine)