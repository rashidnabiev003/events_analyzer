import pandas as pd 
from pathlib import Path

OUTPUT_CSV_TEXT = "src\data\concatenated.csv"

def load_data(path: Path) -> None:
    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise TypeError(f"Error {e}")
    
    df_concat = pd.concat(df[:0], df[:1], axis=1)
    df_concat.to_csv(OUTPUT_CSV_TEXT, index=False, encoding="utf-8")

    return OUTPUT_CSV_TEXT


