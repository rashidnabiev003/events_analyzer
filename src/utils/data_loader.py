# src/utils/data_loader.py
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional

def build_raw(
    xlsx_path: Path,
    column_idx: Tuple[int, ...] = (0, 1, 2),   # (id, title, desc)
    out_path: Path = Path("data/raw.csv"),
    sheet_name: Optional[str] = "Лист2",
    sep: str = " ",
) -> Path:
    df = pd.read_excel(
        xlsx_path,
        header=None,
        engine="openpyxl",
        sheet_name=sheet_name,
    )

    # Индекс последней колонки (берём как "year")
    year_idx = df.shape[1] - 1
    np_name = df.shape[1] - 3

    # Собираем нужные колонки (пропускаем первую строку с шапкой)
    use_cols = [*column_idx, np_name, year_idx]
    # на случай, если последняя колонка вдруг совпала с одной из column_idx
    use_cols = list(dict.fromkeys(use_cols))

    sub = df.iloc[1:, use_cols].copy()

    # Присваиваем имена
    col_names = ["event_id", "c1", "c2", "np_name", "year"]
    sub.columns = col_names

    # event_id — почистим
    sub["event_id"] = (
        sub["event_id"].astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.strip()
    )

    # склеиваем текст
    sub["raw_text"] = (
        sub["c1"].fillna("").astype(str).str.strip()
        + sep +
        sub["c2"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    sub["np_name"] = (
        sub["np_name"].astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.strip()
    )

    # year → просто строка (если столбец есть), иначе пустая строка
    if "year" in sub.columns:
        sub["year"] = sub["year"].astype(str).fillna("").str.strip()
    else:
        sub["year"] = ""

    # оставляем только нужные поля
    sub = sub[["event_id", "raw_text", "np_name",  "year"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    return out_path