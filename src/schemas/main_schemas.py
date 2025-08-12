from pydantic import BaseModel
from typing import Optional, Any, Dict
from typing import List


class OllamaConfig(BaseModel):
    MODEL_NAME: str
    ollama_version: str
    ollama_url: str

class EngineConfig(BaseModel):
    model_path: str
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.8
    quantization: str = 'fp8'
    tensor_parallel_size: int = 1
    max_model_length: int = 2048
    max_batch_size: int = 1000

class AppConfig(BaseModel):
    vllm_engine_config: EngineConfig

class ModelJsonAnswer(BaseModel):
    theme : str
    city : str
    danger_class: str

class Clast(BaseModel):
    region : str
    resources : List[str]

class RiskResp(BaseModel):
    risk: float
    reason: str
    
class Description(BaseModel):
    description: str

class ChatItem(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    json_schema: Any = None


