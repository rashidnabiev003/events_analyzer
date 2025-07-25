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
from pydantic import TypeAdapter
import pandas as pd
import requests
from tqdm import tqdm

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
    xlsx_path: Path,
    lines: int = 100,
    out_csv: Path = RAW_DIR / "enriched.csv",
    sleep_s: float = 0.350,
    flag: int | None = 0
) -> Path:
    """
    1) Берёт первые `lines` строк Excel.
    2) Формирует raw_text (title+desc) → raw.csv.
    3) У каждого raw_text запрашивает Ollama, вытаскивая region/start/end/resources.
    4) Сохраняет enriched.csv.
    """
    # 1-a. raw.csv
    raw_csv = build_raw(xlsx_path, lines)
    df = pd.read_csv(raw_csv)

    # добавляем колонку-заглушки
    for col in ("region", "start", "end", "resources"):
        df[col] = ""

    # 1-b. спрашиваем Ollama по каждой строке
    for idx, row in tqdm(df.iterrows(), 
                        total=len(df),
                        desc="metadata",
                        leave=False):
        prompt = METADATA_PROMPT.format(text=row["raw_text"])
        try:
            ans = chat(prompt=prompt, system_prompt=SYSTEM_METADATA_PROMPT, json_schema=Clast, flag=flag)
            meta = extract_json(ans)
        except Exception as e:
            print("Parse error:", e) 

        df.at[idx, "region"]    = meta['region']
        df.at[idx, "start"]     = meta['start']
        df.at[idx, "end"]       = meta['end']
        df.at[idx, "resources"] = ";".join(meta['resources'])
        time.sleep(sleep_s)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✔ enriched CSV → {out_csv}")
    return out_csv

def risk_matrix(
    enriched_csv: Path = RAW_DIR / "enriched.csv",
    first_col_l: int = 0 ,
    first_col_u: int = 10,
    second_col_l: int = 0,
    second_col_u: int = 50,
    out_csv: Path = RAW_DIR / "risk_matrix.csv",
    global_batch_size: int = 1000
) -> None:
    df = pd.read_csv(enriched_csv)
    targets = df.iloc[first_col_l:first_col_u]

    # Собираем все пары A и B
    all_prompts = []
    all_pairs = []

    for _, row_a in tqdm(targets.iterrows(), total=len(targets), desc="Collecting pairs"):
        a_id = int(row_a.event_id)
        a_text = (
            f"Текст: {row_a.raw_text}\n"
            f"Регион: {row_a.region}; Сроки: {row_a.start}-{row_a.end}; "
            f"Ресурсы: {row_a.resources}"
        )

        # Кандидаты B (исключая A_id)
        candidates = df[df.event_id != a_id].iloc[second_col_l:second_col_u]

        for _, row_b in candidates.iterrows():
            b_id = int(row_b.event_id)
            b_text = (
                f"Текст: {row_b.raw_text}\n"
                f"Регион: {row_b.region}; Сроки: {row_b.start}-{row_b.end}; "
                f"Ресурсы: {row_b.resources}"
            )
            prompt = RISK_PROMPT.format(a_text=a_text, b_text=b_text)
            all_prompts.append(ChatItem(prompt=prompt, system_prompt=SYSTEM_PROMPT))
            all_pairs.append((a_id, b_id))  # ← Сохраняем пары A_id и B_id
    
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


def _parse_cli() -> argparse.Namespace:
    """Возвращает объект с полями .cmd, .webui, .excel, .lines …"""
    p = argparse.ArgumentParser(prog="events_analyzer",
                                description="CLI для build / risk / matrix")

    # глобальный флаг – какой бэкенд использовать
    p.add_argument("--webui",  action="store_true",
                   help="использовать удалённый WebUI (по-умолчанию Ollama)")
    p.add_argument("--ollama", action="store_true",
                   help="форсировать локальный Ollama (переопределяет --webui)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # --- build ---
    sb = sub.add_parser("build", help="собрать enriched.csv из Excel")
    sb.add_argument("excel", type=Path, help="путь к .xlsx")
    sb.add_argument("-l", "--lines", type=int, default=100,
                    help="сколько строк читать (default: 100)")

    # --- matrix ---
    sm = sub.add_parser("matrix", help="матрица рисков N×K")
    sm.add_argument("enriched", type=Path, help="enriched.csv")
    sm.add_argument("-tl", "--first_low", type=int, default=0,
                    help="Нижний порог для первой колонки (default: 0)")
    sm.add_argument("-tu", "--firts_top", type=int, default=10,
                    help="Верхний порог для первой колонки (default: 10)")
    sm.add_argument("-nl", "--second_low",   type=int, default=0,
                    help="Нижний порог для второй колонки (default: 0)")
    sm.add_argument("-nu", "--second_top",   type=int, default=50,
                    help="Верхний порог для второй колонки (default: 50)")
    sm.add_argument("-o", "--out", type=Path, default=RAW_DIR / "risk_matrix.csv",
                    help="путь к выходному файлу (default: data/risk_matrix.csv)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()

    if args.cmd == "build":
        enrich_with_metadata(
            xlsx_path=args.excel,
            lines=args.lines + 1,
        )

    elif args.cmd == "matrix":
        risk_matrix(
            enriched_csv=args.enriched,
            first_col_l=args.first_low,
            first_col_u=args.firts_top,
            second_col_l=args.second_low,
            second_col_u=args.second_top,
            out_csv=args.out,
        )