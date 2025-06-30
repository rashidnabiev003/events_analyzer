import json
import re
import sys
import time
from pathlib import Path
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]   # = C:\Repos\events_analyzer
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any, Dict, List
from src.utils.data_loader import build_raw
from src.schemas.main_schemas import AppConfig, Clast, RiskResp
from pydantic import TypeAdapter
import pandas as pd
import requests
import ollama
import matplotlib.pyplot as plt



CONFIG_PATH = Path("src/configs/confis.json")    


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    txt = Path(path).read_text(encoding="utf-8")
    return TypeAdapter(AppConfig).validate_json(txt)


cfg = load_config()

OLLAMA_URL = cfg.ollama.ollama_url.rstrip("/")        
MODEL_NAME = cfg.ollama.MODEL_NAME                  
RAW_DIR = Path("data")
RAW_DIR.mkdir(exist_ok=True)


SYSTEM_PROMPT = (
    "Ты — ИИ-ассистент, который извлекает структурированные данные из описаний "
    "проектных мероприятий. Отвечай всегда JSON-объектом!"
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
Я планирую два события:

*   **Мероприятие А:** {a_text}
*   **Мероприятие Б:** {b_text}

Предположим, что **мероприятие Б столкнулось со сбоем и отменено/задержано/изменено**.

Проанализируй, **какое влияние (положительное или отрицательное, прямое или косвенное) этот сбой окажет на
выполнение мероприятия А.**  Рассмотри следующие аспекты:

*   Влияние на сроки мероприятия А.
*   Влияние на ресурсы, которые планировались для мероприятия А (финансовые, человеческие, материальные).
*   Влияние на репутацию и восприятие мероприятия А.
*   Оцени вероятность успешного завершения мероприятия А при условии срыва мероприятия Б.
Выдай JSON: {{"risk": <float 0-1>, "reason": "<≤15 слов>"}}.
"""

"""
Ниже описаны два мероприятия.

=== Целевое A (рискует сорваться)
{a_text}

=== Кандидат B (может помешать A)
{b_text}

Оцени, насколько задержка или провал B помешает успешному завершению A.
"""


def _ollama_chat(model: str, prompt: str, schema_model, temperature: float = 0.0):
    """
    Отправляет prompt в Ollama и валидирует ответ с помощью schema_model (Pydantic).
    Возвращает экземпляр schema_model.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        # главное: правильный параметр
        format=schema_model.model_json_schema(),
        options={"temperature": temperature},
    )
    # print(f'{response.message.content}')
    time.sleep(0.35) 
    return response.message.content

def enrich_with_metadata(
    xlsx_path: Path,
    lines: int = 100,
    out_csv: Path = RAW_DIR / "enriched.csv",
    sleep_s: float = 0.2,
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
    for idx, row in df.iterrows():
        prompt = METADATA_PROMPT.format(text=row["raw_text"])
        try:
            ans = _ollama_chat(MODEL_NAME, prompt, Clast)
            meta = json.loads(ans)
        except Exception:
            meta = json.loads({"region": "", "start": "", "end": "", "resources": []})

        # print(meta)

        df.at[idx, "region"]    = meta['region']
        df.at[idx, "start"]     = meta['start']
        df.at[idx, "end"]       = meta['end']
        df.at[idx, "resources"] = ";".join(meta['resources']) 
        print(f"[meta] {idx+1}/{len(df)} ✓")
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
    sleep_s: float = 0.2,
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
            ans = _ollama_chat(MODEL_NAME, prompt, RiskResp)
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
    out_png: Path = RAW_DIR / "risk_heatmap.png",
    sleep_s: float = 0.25,
) -> None:
    """Считает матрицу (target_count × top_n) рисков и сохраняет CSV + PNG."""
    df = pd.read_csv(enriched_csv)

    # ограничиваем список целей A
    targets = df.head(target_count)

    records: List[Dict[str, Any]] = []
    for _, row_a in targets.iterrows():
        a_id   = int(row_a.event_id)
        a_text = (
            f"Текст: {row_a.raw_text}\n"
            f"Регион: {row_a.region}; Сроки: {row_a.start}-{row_a.end}; "
            f"Ресурсы: {row_a.resources}"
        )

        # кандидаты B: первые top_n, кроме самого A
        cand_df = df[df.event_id != a_id].head(top_n)

        for _, row_b in cand_df.iterrows():
            b_id = int(row_b.event_id)
            b_text = (
                f"Текст: {row_b.raw_text}\n"
                f"Регион: {row_b.region}; Сроки: {row_b.start}-{row_b.end}; "
                f"Ресурсы: {row_b.resources}"
            )
            prompt = RISK_PROMPT.format(a_text=a_text, b_text=b_text)

            try:
                ans   = _ollama_chat(MODEL_NAME, prompt, RiskResp)
                meta  = json.loads(ans)
                risk  = float(meta.get("risk", 0.0))
                reason = meta.get("reason", "")
            except Exception:
                risk, reason = 0.0, "parse_error"

            records.append(
                {"A_id": a_id, "B_id": b_id, "risk": risk, "reason": reason}
            )
            print(f"[matrix] A={a_id} → B={b_id}  risk={risk:.2f}")
            time.sleep(sleep_s)

    # 1) CSV
    mat_df = pd.DataFrame(records)
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    mat_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✔ risk matrix CSV → {out_csv}")

    # 2) картинка-heatmap  (pivot-таблица risk)
    pivot = mat_df.pivot(index="A_id", columns="B_id", values="risk").fillna(0)
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, aspect="auto", origin="lower")
    plt.colorbar(label="risk")
    plt.xlabel("B_id (candidates)")
    plt.ylabel("A_id (targets)")
    plt.title("Матрица рисков  Aᵢ ← Bⱼ")
    plt.tight_layout()
    out_png.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✔ Heat-map сохранён → {out_png}")

def _usage() -> None:
    print("Usage:")
    print("  python src/model.py build <excel_path> [lines]")
    print("  python src/model.py risk  <enriched.csv> [target_id] [top_n]")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _usage()

    cmd = sys.argv[1].lower()

    if cmd == "build":
        excel_path = Path(sys.argv[2])
        n_lines = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
        enrich_with_metadata(excel_path, lines=n_lines)

    elif cmd == "risk":
        enriched = Path(sys.argv[2])
        tgt_id = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        top_n_val = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
        risk_vector(enriched, target_id=tgt_id, top_n=top_n_val)

    elif cmd == "matrix":
        enriched     = Path(sys.argv[2])
        tgt_cnt      = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
        top_n_val    = int(sys.argv[4]) if len(sys.argv) >= 5 else 50
        risk_matrix(enriched, target_count=tgt_cnt, top_n=top_n_val)

    else:
        _usage()
