import pandas as pd
from src.utils import data_loader
from pathlib import Path
import tempfile

def test_build_raw_basic(tmp_path):
    # Создаём временный xlsx файл с 3 колонками
    df = pd.DataFrame({0: [1,2,3], 1: ["a","b","c"], 2: ["x","y","z"]})
    xlsx_path = tmp_path / "test.xlsx"
    df.to_excel(xlsx_path, header=False, index=False)
    out_path = tmp_path / "raw.csv"
    result = data_loader.build_raw(xlsx_path=xlsx_path, column_idx=(0,1,2), out_path=out_path, sheet_name=None)
    assert Path(result).exists()
    df2 = pd.read_csv(result)
    assert "event_id" in df2.columns
    assert "raw_text" in df2.columns
    assert df2.shape[0] == 2  # первая строка пропускается (заголовок)

def test_build_raw_empty_file(tmp_path):
    # Пустой DataFrame
    df = pd.DataFrame()
    xlsx_path = tmp_path / "empty.xlsx"
    df.to_excel(xlsx_path, header=False, index=False)
    out_path = tmp_path / "raw_empty.csv"
    try:
        data_loader.build_raw(xlsx_path=xlsx_path, column_idx=(0,1,2), out_path=out_path, sheet_name=None)
    except Exception:
        assert True
    else:
        assert False, "Должно быть исключение на пустом файле"

def test_build_raw_file_not_found():
    import pytest
    with pytest.raises(Exception):
        data_loader.build_raw(xlsx_path="not_exist.xlsx", column_idx=(0,1,2), out_path="out.csv", sheet_name=None) 