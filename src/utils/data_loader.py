from pathlib import Path
import pandas as pd
import csv

def build_raw(
    xlsx_path: Path,
    lines: int = 100,
    column_idx: tuple[int, ...] = (0, 1, 2),          # <-- было (0, 2)
    out_path: Path = Path("data/raw.csv"),
    names: tuple[str, ...] = ("event_id", "c1", "c2"), # <-- было ("event_id", "raw_text")
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

    sub = df.loc[1:lines, column_idx].copy()
    sub.columns = names  # event_id, c1, c2

    # склеиваем текст из колонок 1 и 2
    sub["raw_text"] = (
        sub["c1"].fillna("").astype(str).str.strip() + " " +
        sub["c2"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # минимальная очистка event_id
    sub["event_id"] = (
        sub["event_id"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.strip()
    )

    # оставляем только нужные 2 колонки
    sub = sub[["event_id", "raw_text"]]

    # сохраняем
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    return out_path