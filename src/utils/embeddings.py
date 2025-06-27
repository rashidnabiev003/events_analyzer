import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TEXT_COLUMN = 0
EMB_MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_EMB = r"src\data\data_col0.csv"
OUTPUT_NPY = r"src\data\data_bytes.npy"

def text_to_embeddings(df) -> str:
    texts = df.iloc[:, TEXT_COLUMN].astype(str).fillna("").tolist()
    model = SentenceTransformer(EMB_MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    np.save(OUTPUT_NPY, embeddings)

    return f"Ready and saved in {OUTPUT_NPY}"
