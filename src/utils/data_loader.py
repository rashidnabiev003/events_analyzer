# src/utils/data_loader.py
from pathlib import Path
import pandas as pd
import csv

def build_raw(
    xlsx_path: Path,
    lines: int = 100,
    column_idx: int = 1,
    out_path: Path = Path("data/raw.csv"),
    sheet_name: str | None = "Лист2"
) -> Path:
    # --- читаем только нужное кол-во строк -----------------------------
    df = pd.read_excel(
        xlsx_path,
        header=None,          # нет шапки – берём всё как есть
        engine="openpyxl",
        nrows=lines,
        sheet_name=sheet_name,
    )

    if column_idx >= df.shape[1]:
        raise ValueError(f"В листе только {df.shape[1]} колонок, а запрошена {column_idx}")

    # --- колонка → Series ---------------------------------------------
    raw_series = (
        df.iloc[:, column_idx]        # берём весь столбец
          .dropna()                   # PRAVKA №1: убираем NaN, пока они ещё NaN
          .astype(str)
          .str.replace(r"[\r\n]+", " ", regex=True)   # PRAVKA №2: перевод строки → пробел
          .str.strip()
    ).head(lines)                     # лишних уже нет, но пусть останется страховка

    # --- собираем датафрейм -------------------------------------------
    out_df = (raw_series
              .reset_index(drop=True)
              .rename("raw_text")
              .to_frame()
              .reset_index(names="event_id"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8",
                  quoting=csv.QUOTE_MINIMAL)          # корректные кавычки для Excel
    return out_path
