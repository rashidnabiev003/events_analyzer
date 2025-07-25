import pytest
from src.vllm_engine import VLLMEngine, extract_json
from src.schemas.main_schemas import EngineConfig, ChatItem, RiskResp
from pydantic import BaseModel
import re
import json

class DummyLLM:
    def chat(self, prompts, sampling_params):
        # Возвращает фиктивный ответ с json
        class Output:
            def __init__(self, text):
                self.outputs = [type('O', (), {'text': text})()]
        return [Output('{"risk": 0.5, "reason": "ok"}') for _ in prompts]

class DummyEngineConfig(EngineConfig):
    model_path: str = "dummy"
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.8
    quantization: str = 'fp8'
    tensor_parallel_size: int = 1
    max_model_length: int = 1024
    max_batch_size: int = 2

def test_extract_json():
    text = '<think>...</think>```json {"risk": 0.1, "reason": "test"} ```'
    result = extract_json(text)
    assert result == {"risk": 0.1, "reason": "test"}

def test_vllm_engine_chat_batch(monkeypatch):
    # Подменяем LLM на DummyLLM
    import src.vllm_engine as vllm_engine_mod
    vllm_engine_mod.LLM = lambda *a, **kw: DummyLLM()
    config = DummyEngineConfig()
    engine = VLLMEngine(engine_config=config, system_prompt="sys")
    items = [ChatItem(prompt="hi", system_prompt="sys") for _ in range(2)]
    result = engine.chat_batch(items=items, json_schema=RiskResp, max_retries=1)
    assert isinstance(result, list)
    assert all("risk" in r and "reason" in r for r in result)
    assert result[0]["risk"] == 0.5 