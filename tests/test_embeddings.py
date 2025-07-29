import pytest
import pandas as pd
from src.utils import embeddings
import numpy as np

def test_text_to_embeddings(monkeypatch, tmp_path):
    # Подменяем SentenceTransformer и np.save
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: type("M", (), {"encode": lambda self, texts, **kw: np.zeros((len(texts), 3))})())
    monkeypatch.setattr(embeddings.np, "save", lambda path, arr: None)
    df = pd.DataFrame({0: ["a", "b"]})
    result = embeddings.text_to_embeddings(df)
    assert "Ready and saved" in result 


def test_text_to_embeddings_empty(monkeypatch):
    from src.utils import embeddings
    import pandas as pd
    df = pd.DataFrame()
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: type("M", (), {"encode": lambda self, texts, **kw: []})())
    monkeypatch.setattr(embeddings.np, "save", lambda path, arr: None)
    result = embeddings.text_to_embeddings(df)
    assert "empty" in result.lower() or result == "" 