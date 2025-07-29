from src.schemas import main_schemas

def test_ollama_config():
    c = main_schemas.OllamaConfig(MODEL_NAME="m", ollama_version="1.0", ollama_url="url")
    assert c.MODEL_NAME == "m"
    assert c.ollama_url == "url"

def test_engine_config():
    c = main_schemas.EngineConfig(model_path="m")
    assert c.model_path == "m"
    assert c.trust_remote_code is True
    assert c.gpu_memory_utilization == 0.8
    assert c.quantization == 'fp8'
    assert c.tensor_parallel_size == 1
    assert c.max_model_length == 2048
    assert c.max_batch_size == 1000

def test_app_config():
    ec = main_schemas.EngineConfig(model_path="m")
    ac = main_schemas.AppConfig(vllm_engine_config=ec)
    assert ac.vllm_engine_config.model_path == "m"

def test_clast():
    c = main_schemas.Clast(region="r", start="s", end="e", resources=["a"])
    assert c.region == "r"
    assert c.resources == ["a"]

def test_model_json_answer():
    m = main_schemas.ModelJsonAnswer(theme="t", city="c", danger_class="d")
    assert m.theme == "t"
    assert m.city == "c"
    assert m.danger_class == "d"

def test_risk_resp():
    r = main_schemas.RiskResp(risk=0.5, reason="test")
    assert r.risk == 0.5
    assert r.reason == "test"

def test_description():
    d = main_schemas.Description(description="desc")
    assert d.description == "desc"

def test_chat_item():
    ci = main_schemas.ChatItem(prompt="p", system_prompt="s", json_schema=dict)
    assert ci.prompt == "p"
    assert ci.system_prompt == "s"
    assert ci.json_schema == dict 