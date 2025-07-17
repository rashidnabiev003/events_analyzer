import json
import re
import httpx
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
from src.utils.data_loader import build_raw
from src.schemas.main_schemas import (AppConfig,
                                        Clast,
                                        RiskResp,
                                        Description)
from pydantic import TypeAdapter
import pandas as pd
import requests
import ollama
from tqdm import tqdm

CONFIG_PATH = CONFIG_PATH = ROOT / "src" / "configs" / "config.json" 
def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    txt = Path(path).read_text(encoding="utf-8")
    return TypeAdapter(AppConfig).validate_json(txt)


cfg = load_config()
WEBUI_KEY = os.getenv("WEBUI_API_KEY", "")
OLLAMA_URL = cfg.ollama.ollama_url.rstrip("/")        
MODEL_NAME = cfg.ollama.MODEL_NAME 
WEBUI_MODEL_NAME = cfg.webui_ollama.MODEL_NAME      
WEBUI_URL = cfg.webui_ollama.ollama_url.rstrip("/")
RAW_DIR = Path("data")
RAW_DIR.mkdir(exist_ok=True)
TEMPERATURE = 0.3
TIMEOUT = 180


SYSTEM_PROMPT = (
   """
/no_think
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
{{
"risk":"probability_failure"
"reason": "
Исходное мероприятие: <text>

Наиболее связанное мероприятие: <text>

Объяснение зависимости: <dependency>"}}
Начинайте сразу после этого сообщения.
"""


json_re = re.compile(r"\{[\s\S]*\}")

def extract_json(text: str) -> dict:
    # 1. убираем <think>…</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)

    # 2. удаляем лишь маркеры ```json и ```
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.I)

    # 3. ищем первый {...}
    m = json_re.search(text)
    if not m:
        raise ValueError("JSON not found in model output")
    return json.loads(m.group(0))

def chat(prompt: str,          
    system_prompt: str | None = SYSTEM_PROMPT,
    json_schema=None,
    flag=1):

    if flag == 0:
        headers = {"Content-Type": "application/json"}
        if WEBUI_KEY:
            headers["Authorization"] = f"Bearer {WEBUI_KEY}"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
        "model": WEBUI_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "temperature": TEMPERATURE,
        "max_tokens": 1500,
        }

        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_object"
                }
        
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.post(f"{WEBUI_URL}/api/chat/completions",
                        headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    else:
        response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        # главное: правильный параметр
        format=json_schema.model_json_schema(),
        options={"temperature": TEMPERATURE},
        )
        time.sleep(0.35) 
        return response.message.content

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
            ans = chat(prompt=prompt, json_schema=Clast, flag=flag)
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

def risk_vector(
    enriched_csv: Path,
    target_id: int = 0,
    top_n: int = 100,
    out_csv: Path = RAW_DIR / "risk_vector.csv",
    sleep_s: float = 0.350,
    flag:int | None = 0,
) -> Path:
    """
    Генерирует CSV:
        target_id, candidate_id, risk, reason
    """
    df = pd.read_csv(enriched_csv)

    if target_id not in df["event_id"].values:
        raise ValueError(f"event_id={target_id} не найден в {enriched_csv}")

    # Формируем A-текст (включая метаданные)
    row_a = df[df.event_id == target_id].iloc[0]
    a_text = (
        f"Текст: {row_a.raw_text}\n"
        f"Регион: {row_a.region}; Сроки: {row_a.start}-{row_a.end}; "
        f"Ресурсы: {row_a.resources}"
    )

    # Кандидаты: все, кроме A, возьмём top_n
    cand_df = df[df.event_id != target_id].head(top_n)
    records: List[Dict[str, Any]] = []

    for i, row_b in cand_df.iterrows():
        b_text = (
            f"Текст: {row_b.raw_text}\n"
            f"Регион: {row_b.region}; Сроки: {row_b.start}-{row_b.end}; "
            f"Ресурсы: {row_b.resources}"
        )
        prompt = RISK_PROMPT.format(a_text=a_text, b_text=b_text)

        try:
            ans = chat(MODEL_NAME, prompt, RiskResp, flag=flag)
            meta = json.loads(ans)
            risk = float(meta.get("risk", 0.0))
            reason = meta.get("reason", "")
        except Exception:
            risk, reason = 0.0, "parse_error"

        records.append(
            {
                "target_id": target_id,
                "candidate_id": int(row_b.event_id),
                "risk": risk,
                "reason": reason,
            }
        )
        print(f"[risk] {len(records)}/{top_n}  risk={risk:.2f}")
        time.sleep(sleep_s)

    out_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✔ risk vector CSV → {out_csv}")
    return out_csv


def risk_matrix(
    enriched_csv: Path,
    target_count: int = 10,          # сколько A-мероприятий взять (0 … target_count-1)
    top_n: int = 50,                 # для каждого A сколько кандидатов B
    out_csv: Path = RAW_DIR / "risk_matrix.csv",
    sleep_s: float = 0.25,
    flag: int | None = None
) -> None:
    """Считает матрицу (target_count × top_n) рисков и сохраняет CSV + PNG."""
    df = pd.read_csv(enriched_csv)

    # ограничиваем список целей A
    targets = df.head(target_count)

    records: List[Dict[str, Any]] = []
    for _, row_a in tqdm(targets.iterrows(),
                         total=len(targets),
                         leave=False,
                         desc="matrix"):
        a_id   = int(row_a.event_id)
        a_text = (
            f"Текст: {row_a.raw_text}\n"
            f"Регион: {row_a.region}; Сроки: {row_a.start}-{row_a.end}; "
            f"Ресурсы: {row_a.resources}"
        )

        # кандидаты B: первые top_n, кроме самого A
        cand_df = df[df.event_id != a_id].head(top_n)

        for _, row_b in tqdm(cand_df.iterrows(),
                             total=len(cand_df),
                             desc=f"A = {a_id}",
                             leave=False):
            b_id = int(row_b.event_id)
            b_text = (
                f"Текст: {row_b.raw_text}\n"
                f"Регион: {row_b.region}; Сроки: {row_b.start}-{row_b.end}; "
                f"Ресурсы: {row_b.resources}"
            )
            prompt = RISK_PROMPT.format(a_text=a_text, b_text=b_text)

            try:
                ans = chat(prompt, system_prompt=SYSTEM_PROMPT,json_schema=RiskResp, flag=flag)
                meta  = extract_json(ans)
                risk  = float(meta.get("risk", 0.0))
                reason = meta.get("reason", "")
            except Exception as e:
                print("parse_error:", e)

            records.append(
                {"A_id": a_id, "B_id": b_id, "risk": risk, "reason": reason}
            )
            time.sleep(sleep_s)

    # 1) CSV
    mat_df = pd.DataFrame(records)
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    mat_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✔ risk matrix CSV → {out_csv}")


def explainer(xlsx_path:Path,
              sleep_s: float = 0.25,
              flag:int | None = None,
              out_csv: Path | None = None,
              sheet_name:str | None = "Лист1"):
    
    df = pd.read_excel(
        xlsx_path,
        header=None,          
        engine="openpyxl",
        sheet_name=sheet_name,
    )
    
    records : List[Dict[str, Any]] = []
    column_idx = (1, 2, 4 , 5, 6)
    names = ("Наименование B", "Описание B", "Наименование А", "Описание А", "Риск")

    sub = df.loc[1:, column_idx].copy()
    sub.columns = names

    for _, raw in  tqdm(sub.iterrows(),
                  total=len(sub),
                  leave=False):
        b_name = raw["Наименование B"]
        b_text = raw["Описание B"]
        a_name = raw["Наименование А"]
        a_text = raw["Описание А"]
        risk = raw["Риск"]

        prompt =    THINK_PROMPT.format(a_text=a_text, b_text=b_text, a_name=a_name, b_name=b_name, risk=risk)
        try:
            ans = chat(prompt, system_prompt=THINK_SYSTEM_PROMPT,json_schema=Description, flag=flag)
            meta  = extract_json(ans)
            reason = meta.get("description", "")
        except Exception as e:
            print("parse_error:", e)

        time.sleep(sleep_s)
        records.append(reason)
    
    if out_csv is None:
        out_csv = Path("data/explained.csv")
    mat_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mat_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✔ explanation CSV → {out_csv}")




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

    # --- risk ---
    sr = sub.add_parser("risk", help="вектор рисков 1×N")
    sr.add_argument("enriched", type=Path, help="enriched.csv")
    sr.add_argument("-t", "--target-id", type=int, default=0,
                    help="event_id мероприятия A (default: 0)")
    sr.add_argument("-n", "--top", type=int, default=100,
                    help="сколько кандидатов B (default: 100)")

    # --- matrix ---
    sm = sub.add_parser("matrix", help="матрица рисков N×K")
    sm.add_argument("enriched", type=Path, help="enriched.csv")
    sm.add_argument("-c", "--count", type=int, default=10,
                    help="сколько A-мероприятий (default: 10)")
    sm.add_argument("-n", "--top",   type=int, default=50,
                    help="сколько B-кандидатов на каждое A (default: 50)")

    se = sub.add_parser("explain", help="генерирует текстовое объяснение зависимости A←B")
    se.add_argument("excel", type=Path, help="xlsx с колонками (B_name,B_desc,A_name,A_desc, risk)")
    se.add_argument("--sheet", default="Лист1", help="имя листа Excel (default: Лист1)")
    se.add_argument("--out",   type=Path, default="data/explained.csv",
                   help="путь вывода CSV (default: data/explained.csv)")
    se.add_argument("--sleep", type=float, default=0.25, help="пауза между вызовами (сек)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()

    # выбор бэкенда: 0 = WebUI, 1 = Ollama
    flag = 1                             # по-умолчанию Ollama
    if args.webui and not args.ollama:
        flag = 0

    if args.cmd == "build":
        enrich_with_metadata(
            xlsx_path=args.excel,
            lines=args.lines + 1,
            flag=flag,
        )

    elif args.cmd == "risk":
        risk_vector(
            enriched_csv=args.enriched,
            target_id=args.target_id,
            top_n=args.top,
            flag=flag,
        )

    elif args.cmd == "matrix":
        risk_matrix(
            enriched_csv=args.enriched,
            target_count=args.count,
            top_n=args.top,
            flag=flag,
        )
    elif args.cmd == "explain":
        explainer(
            xlsx_path=args.excel,
            sheet_name=args.sheet,
            out_csv=args.out,
            sleep_s=args.sleep,
            flag=flag,
        )
