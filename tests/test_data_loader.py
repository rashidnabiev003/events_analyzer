import pandas as pd
from src.utils import data_loader
from pathlib import Path
import tempfile

def test_build_raw(tmp_path):
    # Создаём временный xlsx файл
    df = pd.DataFrame({0: [1,2,3], 2: ["a","b","c"]})
    xlsx_path = tmp_path / "test.xlsx"
    df.to_excel(xlsx_path, header=False, index=False)
    out_path = tmp_path / "raw.csv"
    result = data_loader.build_raw(xlsx_path=xlsx_path, lines=2, column_idx=(0,2), out_path=out_path, names=("event_id","raw_text"), sheet_name=None)
    assert Path(result).exists()
    df2 = pd.read_csv(result)
    assert "event_id" in df2.columns
    assert "raw_text" in df2.columns 