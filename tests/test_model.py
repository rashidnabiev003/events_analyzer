import pytest
from src import model
from pathlib import Path
import pandas as pd
import tempfile
import os

def test_extract_json():
    text = '<think>...</think>```json {"risk": 0.2, "reason": "ok"} ```'
    result = model.extract_json(text)
    assert result == {"risk": 0.2, "reason": "ok"}

def test_enrich_with_metadata(monkeypatch):
    # Подменяем chat и extract_json на фиктивные
    monkeypatch.setattr(model, "chat", lambda *a, **kw: '{"region": "A", "start": "2020", "end": "2021", "resources": ["r1"]}')
    monkeypatch.setattr(model, "extract_json", lambda x: {"region": "A", "start": "2020", "end": "2021", "resources": ["r1"]})
    # Создаём временный raw.csv
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_csv = Path(tmpdir) / "raw.csv"
        df = pd.DataFrame({"event_id": [1], "raw_text": ["text"]})
        df.to_csv(raw_csv, index=False)
        # monkeypatch путь внутри функции
        monkeypatch.setattr(model, "RAW_DIR", Path(tmpdir))
        out_csv = model.enrich_with_metadata(xlsx_path=None, lines=1, out_csv=raw_csv, sleep_s=0, flag=0)
        assert Path(out_csv).exists()
        df2 = pd.read_csv(out_csv)
        assert "region" in df2.columns

def test_risk_matrix(monkeypatch):
    # Готовим фиктивный enriched.csv
    with tempfile.TemporaryDirectory() as tmpdir:
        enriched = Path(tmpdir) / "enriched.csv"
        df = pd.DataFrame({"event_id": [1,2], "raw_text": ["a","b"], "region": ["A","B"], "start": ["2020","2021"], "end": ["2021","2022"], "resources": ["r1","r2"]})
        df.to_csv(enriched, index=False)
        # Подменяем engine.chat_batch
        class DummyEngine:
            def chat_batch(self, *a, **kw):
                return [{"risk": 0.1, "reason": "ok"} for _ in range(1)]
        monkeypatch.setattr(model, "engine", DummyEngine())
        monkeypatch.setattr(model.cfg, "vllm_engine_config", type("C", (), {"max_batch_size": 10})())
        out_csv = Path(tmpdir) / "risk_matrix.csv"
        model.risk_matrix(enriched_csv=enriched, first_col_l=0, first_col_u=1, second_col_l=1, second_col_u=2, out_csv=out_csv)
        assert out_csv.exists()
        df2 = pd.read_csv(out_csv)
        assert "risk" in df2.columns 


def test_extract_json_invalid():
    from src import model
    # Некорректный JSON
    text = '<think>...</think>```json {risk: 0.2, reason: ok} ```'
    result = model.extract_json(text)
    assert result is None or result == {}

def test_enrich_with_metadata_missing_columns(monkeypatch, tmp_path):
    from src import model
    import pandas as pd
    # Подменяем chat и extract_json на фиктивные
    monkeypatch.setattr(model, "chat", lambda *a, **kw: '{}')
    monkeypatch.setattr(model, "extract_json", lambda x: {})
    # Создаём временный raw.csv без нужных колонок
    raw_csv = tmp_path / "raw.csv"
    df = pd.DataFrame({"wrong_col": [1], "text": ["text"]})
    df.to_csv(raw_csv, index=False)
    monkeypatch.setattr(model, "RAW_DIR", tmp_path)
    try:
        model.enrich_with_metadata(xlsx_path=None, lines=1, out_csv=raw_csv, sleep_s=0, flag=0)
    except Exception:
        assert True
    else:
        assert False, "Должно быть исключение при отсутствии нужных колонок" 