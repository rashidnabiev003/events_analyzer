from pydantic import BaseModel
from typing import List


class OllamaConfig(BaseModel):
    MODEL_NAME: str
    ollama_version: str
    ollama_url: str

class AppConfig(BaseModel):
    ollama: OllamaConfig
    webui_ollama : OllamaConfig

class ModelJsonAnswer(BaseModel):
    theme : str
    city : str
    danger_class: str

class Clast(BaseModel):
    region : str
    start: str
    end: str
    resources : List[str]

class RiskResp(BaseModel):
    risk: float
    reason: str
    
