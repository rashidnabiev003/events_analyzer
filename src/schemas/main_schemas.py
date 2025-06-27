from pydantic import BaseModel


class OllamaConfig(BaseModel):
    MODEL_NAME: str
    ollama_version: str
    ollama_url: str

class AppConfig(BaseModel):
    ollama: OllamaConfig

class ModelJsonAnswer(BaseModel):
    theme : str
    city : str
    danger_class: str
