import pandas as pd
import numpy as np
import pytest
from src.utils.embeddings_search import EventsSemanticSearch, BuildConfig
from pathlib import Path

def test_build_from_dataframe(tmp_path):
    # Минимальный DataFrame для индекса
    df = pd.DataFrame({
        "event_id": [1, 2],
        "raw_text": ["test event one", "test event two"]
    })
    workdir = tmp_path / "search"
    search = EventsSemanticSearch(workdir=workdir, cfg=BuildConfig(batch_size=2, force_cpu=True))
    search.build_from_dataframe(df)
    assert (workdir / "events.faiss").exists()
    assert (workdir / "metadata.json").exists()
    assert (workdir / "embeddings.npy").exists()

    # Проверка загрузки индекса
    search.load()
    assert search.index is not None
    assert isinstance(search.metadata, list)

# Тест на пустой DataFrame

def test_build_from_empty_dataframe(tmp_path):
    df = pd.DataFrame({"event_id": [], "raw_text": []})
    workdir = tmp_path / "search_empty"
    search = EventsSemanticSearch(workdir=workdir, cfg=BuildConfig(batch_size=2, force_cpu=True))
    with pytest.raises(Exception):
        search.build_from_dataframe(df) 