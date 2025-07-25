from src.schemas import main_schemas

def test_ollama_config():
    c = main_schemas.OllamaConfig(MODEL_NAME="m", ollama_version="1.0", ollama_url="url")
    assert c.MODEL_NAME == "m"
    assert c.ollama_url == "url"

def test_engine_config():
    c = main_schemas.EngineConfig(model_path="m")
    assert c.model_path == "m"

def test_app_config():
    ec = main_schemas.EngineConfig(model_path="m")
    ac = main_schemas.AppConfig(vllm_engine_config=ec)
    assert ac.vllm_engine_config.model_path == "m"

def test_clast():
    c = main_schemas.Clast(region="r", start="s", end="e", resources=["a"]) 
    assert c.region == "r"
    assert c.resources == ["a"] 