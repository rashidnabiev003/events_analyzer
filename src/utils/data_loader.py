# src/utils/data_loader.py
from pathlib import Path
import pandas as pd

def build_raw(
    xlsx_path: Path,
    column_idx: tuple[int, ...] = (0, 1, 2),
    out_path: Path = Path("data/raw.csv"),
    sheet_name: str | None = "Лист1",
    sep: str = " ",   # разделитель между 1 и 2 колонками
) -> Path:
    df = pd.read_excel(
        xlsx_path,
        header=None,
        engine="openpyxl",
        sheet_name=sheet_name,
    )

    # берём строки 1..lines включительно 
    sub = df.iloc[1: , list(column_idx)].copy()
    sub.columns = ["event_id", "c1", "c2"]

    # чистим и склеиваем
    sub["event_id"] = (
        sub["event_id"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.strip()
    )

    sub["raw_text"] = (
        sub["c1"].fillna("").astype(str).str.strip()
        + sep +
        sub["c2"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # оставляем только нужные столбцы
    sub = sub[["event_id", "raw_text"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    return out_path