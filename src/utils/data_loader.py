# src/utils/data_loader.py
from pathlib import Path
import pandas as pd
import csv

def build_raw(
    xlsx_path: Path,
    lines: int = 100,
    column_idx: tuple[int, ...] = (0, 2),
    out_path: Path = Path("data/raw.csv"),
    names: tuple[str, ...] = ("event_id", "raw_text"),
    sheet_name: str | None = "Лист3"
) -> Path:
    # --- читаем только нужное кол-во строк -----------------------------
    df = pd.read_excel(
        xlsx_path,
        header=None,          
        engine="openpyxl",
        nrows=lines,
        sheet_name=sheet_name,
    )

    sub = df.loc[1: lines, column_idx].copy()
    sub.columns = names

    # минимальная очистка: убираем переносы строк и лишние пробелы
    for col in names:
        sub[col] = (
            sub[col]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

    # сохраняем
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
