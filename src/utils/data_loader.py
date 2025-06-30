# src/utils/data_loader.py
from pathlib import Path
import pandas as pd

def build_raw_csv(xlsx_path: Path, lines: int = 100) -> Path:
    """
    Берёт Excel, конкатенирует колонку 0 + колонку 1,
    сохраняет data/raw.csv и возвращает путь.
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl").head(lines)
    df["raw_text"] = (
        df.iloc[:, 0].astype(str).str.strip() + ". " +
        df.iloc[:, 1].astype(str).str.strip()
    )
    out = Path("data/raw.csv")
    out.parent.mkdir(exist_ok=True, parents=True)
    df.reset_index(names="event_id")[["event_id", "raw_text"]].to_csv(
        out, index=False, encoding="utf-8"
    )
    return out


def build_raw(
    xlsx_path: Path,
    lines: int = 100,
    column_idx: int = 1,                 # ←  1  = «вторая колонка»
    out_path: Path = Path("data/raw_2.csv"),
) -> Path:
    """
    Берёт Excel, независимо от заголовков, вытаскивает column_idx-ю колонку,
    обрезает до `lines`, сохраняет CSV (event_id, raw_text).
    """
    # читаем лист целиком как DataFrame, без заголовков
    df = pd.read_excel(xlsx_path, header=None, engine="openpyxl")

    # проверяем наличие запрошенной колонки
    n_cols = df.shape[1] if isinstance(df, pd.DataFrame) else 1
    if column_idx >= n_cols:
        raise ValueError(
            f"В листе только {n_cols} колонок, а запрошена колонка {column_idx}"
        )

    # берём нужную колонку → Series, чистим пробелы, берём первые `lines`
    raw_series = (
        df.iloc[:lines, column_idx]
        .astype(str)
        .str.strip()
    )

    # формируем DataFrame для сохранения
    out_df = (
        raw_series.reset_index(drop=True)
                  .rename("raw_text")
                  .to_frame()
                  .reset_index(names="event_id")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path
