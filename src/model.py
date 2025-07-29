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
from typing import Any, Dict, List
from src.vllm_engine import VLLMEngine
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


SYSTEM_PROMPT = (
   """
Ты — эксперт по анализу рисков в проектах национальных мероприятий. Перед тобой пара событий:
- Мероприятие исходное
- Мероприятие наиболее связное
Предположим, что связное столкнулось со сбоем (отменено, задержано или существенно изменено). Ваша задача — по строгой методологии оценить влияние этого сбоя на A и выдать результат в виде JSON, где ключевой показатель — вероятность негативного исхода.

1. Оцените уровень влияния по трём аспектам и дайте каждой оценке дискретное значение от 0 до 4, где:  
   - 0 = отсутствие влияния  
   - 1 = незначительное  
   - 2 = умеренное  
   - 3 = значительное  
   - 4 = критическое

   Аспекты:  
   - time_level — влияние на сроки  
   - resource_level — влияние на ресурсы (финансовые, человеческие, материальные)   

2. Вычислите общий риск как сумму 2 уровней:  
total_risk = time_level + resource_level // диапазон 0–12

3. Вычислите вероятность негативного исхода как отношение:  
probability_failure = total_risk / 12.0 // число от 0.0 до 1.0

**Формат ответа (строго JSON):**
{
  "risk": probability_failure,
  "reason": "Исходное мероприятие: ...\nНаиболее связанное мероприятие: ...\nОбъяснение зависимости: ..."
}
"""
)
SYSTEM_METADATA_PROMPT = (
"""
Ты помощник по извлечению метаданных из мероприятия
Извлеки JSON с полями:
region  – регион (строка),
start   – дата/квартал начала (строка),
end     – дата/квартал окончания (строка),
resources – список ключевых ресурсов или подрядчиков (массив строк).
"""
)

METADATA_PROMPT = """
Извлеки JSON с полями:
region  – регион (строка),
start   – дата/квартал начала (строка),
end     – дата/квартал окончания (строка),
resources – список ключевых ресурсов или подрядчиков (массив строк).

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
engine = VLLMEngine(engine_config=cfg.vllm_engine_config, system_prompt=SYSTEM_PROMPT) 
RAW_DIR = Path("data")
RAW_DIR.mkdir(exist_ok=True)

def enrich_with_metadata(
    df: pd.DataFrame,
    vllm_engine: VLLMEngine,
    json_schema: Any
) -> pd.DataFrame:
    """
    Обогащает DataFrame df метаданными через пакетные запросы vllm.
    """
    # Prepare placeholder columns
    for col in ("region", "start", "end", "resources"):
        df[col] = ""

    # Build chat items for all rows
    items = [
        ChatItem(
            system_prompt=SYSTEM_METADATA_PROMPT,
            prompt=METADATA_PROMPT.format(text=raw_text)
        ) for raw_text in df["raw_text"].tolist()
    ]

    # Batch request to VLLMEngine
    responses = vllm_engine.chat_batch(
        items=items,
        json_schema=json_schema
    )

    # Populate DataFrame
    for idx, resp in enumerate(responses):
        df.at[idx, "region"] = resp.get("region", "")
        df.at[idx, "start"] = resp.get("start", "")
        df.at[idx, "end"] = resp.get("end", "")
        resources = resp.get("resources", [])
        df.at[idx, "resources"] = ";".join(resources) if isinstance(resources, list) else resources

    return df

def risk_matrix(
    enriched_csv: Path = RAW_DIR / "enriched.csv",
    first_col_l: int = 0 ,
    first_col_u: int = 10,
    second_col_l: int = 0,
    second_col_u: int = 50,
    out_csv: Path = RAW_DIR / "risk_matrix.csv",
    candidate_pairs: Optional[List[tuple[int, int]]] = None,
    searcher: Optional[EventsSemanticSearch] = None
) -> None:
    df = pd.read_csv(enriched_csv)
    targets = df.iloc[first_col_l:first_col_u]

    if candidate_pairs is None and searcher is not None:
        candidate_pairs = searcher.similar_pairs_for_all(df, text_col="raw_text", id_col="event_id", top_k_per_event=20)

    if not candidate_pairs:
        targets = df.iloc[first_col_l:first_col_u]
        candidates = df.iloc[second_col_l:second_col_u]
        candidate_pairs = [
            (str(a.event_id), str(b.event_id))
            for _, a in targets.iterrows()
            for _, b in candidates.iterrows()
            if str(a.event_id) != str(b.event_id)
        ]

    # Собираем все пары A и B
    all_prompts = []
    all_pairs = []
    id_to_row = {str(r.event_id): r for _, r in df.iterrows()}
    from src.schemas.main_schemas import ChatItem  # уже есть у вас

    for a_id, b_id in candidate_pairs:
        row_a, row_b = id_to_row[a_id], id_to_row[b_id]
        a_text = f"Текст: {row_a.raw_text}\nРегион: {row_a.region}; Сроки: {row_a.start}-{row_a.end}; Ресурсы: {row_a.resources}"
        b_text = f"Текст: {row_b.raw_text}\nРегион: {row_b.region}; Сроки: {row_b.start}-{row_b.end}; Ресурсы: {row_b.resources}"
        prompt = RISK_PROMPT.format(a_text=a_text, b_text=b_text)  # у вас уже определён
        all_prompts.append(ChatItem(prompt=prompt, system_prompt=SYSTEM_PROMPT))
        all_pairs.append((int(a_id), int(b_id)))
    
    responses = []
    for i in range(0, len(all_prompts), cfg.vllm_engine_config.max_batch_size):
        batch = all_prompts[i:i + cfg.vllm_engine_config.max_batch_size]
        batch_responses = engine.chat_batch(batch, json_schema=RiskResp)
        responses.extend(batch_responses)
    
    # Записываем результаты в CSV
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["A_id", "B_id", "risk", "reason"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for (a_id, b_id), response in zip(all_pairs, responses):
            record = {
                "A_id": a_id,
                "B_id": b_id,
                "risk": float(response.get("risk", 0.0)),
                "reason": response.get("reason", "")
            }
            writer.writerow(record)
            
    print(f"✔ risk matrix → {out_csv}")

def get_xlsx_files(input_dir: Path) -> list[Path]:
    """
    Возвращает список всех .xlsx файлов в папке input_dir.
    """
    return sorted(input_dir.glob("*.xlsx"))

def process_file(
    xlsx_path: Path,
    output_dir: Path,
    vllm_engine: VLLMEngine | None = engine
) -> None:
    name = xlsx_path.stem
    raw_csv = build_raw(
        xlsx_path=xlsx_path,
        column_idx=(0, 1, 2),
        out_path=output_dir / "raw.csv",
    )
    df = pd.read_csv(raw_csv)
    # Enrich metadata

    search_dir = output_dir / "search_index"
    searcher = EventsSemanticSearch(workdir=search_dir, cfg=BuildConfig(force_cpu=False), enable_rerank=True)
    searcher.build_from_dataframe(df)

    enriched_df = enrich_with_metadata(
        df=df,
        vllm_engine=vllm_engine,
        json_schema=Clast
    )
    enriched_csv = output_dir / f"enriched_{name}.csv"
    enriched_df.to_csv(enriched_csv, index=False)
    print(f"✔ enriched CSV → {enriched_csv}")

    pairs = searcher.similar_pairs_for_all(enriched_df, text_col="raw_text", id_col="event_id", top_k_per_event=20)
    
    # Risk matrix
    risk_csv = output_dir / f"risk_{name}.csv"
    risk_matrix(
        enriched_csv=enriched_csv,
        first_col_l=0, first_col_u=len(enriched_df),
        second_col_l=0, second_col_u=len(enriched_df),
        out_csv=risk_csv,
        candidate_pairs=pairs,
        searcher=searcher
    )
    print(f"✔ processed {name} → {risk_csv}")

def process_folder(
    input_dir: Path,
    output_dir: Path
) -> None:
    """
    Обходит все xlsx-файлы в input_dir и обрабатывает каждый через process_file().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for xlsx_path in get_xlsx_files(input_dir):
        process_file(
            xlsx_path=xlsx_path,
            output_dir=output_dir
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
    # Инициализация движка VLLM 
    vllm_engine = engine

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_path.is_dir():
        process_folder(args.input_path, args.output_dir)
    else:
        process_file(args.input_path, args.output_dir, vllm_engine)