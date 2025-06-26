import ollama
import re 
import json
from typing  import Dict, List, Any
from ..schemas.main_schemas import AppConfig



def load_config(path: str = r"C:\Repos\events_analyzer\configs\confis.json") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AppConfig(**data)

cfg = load_config()
MODEL = cfg.ollama.MODEL_NAME
OLLAMA_URL = cfg.ollama.ollama_url.rstrip("/")
MODEL_OPTIONS = {"temperature": 0.10, "repeat_penalty": 1.3, "top_p": 0.9, "num_ctx": 8046, "grammar": "ru"}

SYSTEM_PROMPT = ("")
USER_PROMPT = ("")
FEWSHOT_SUMMARY = ("")

class Ollama_chat:
    def __init__(self, model:str = MODEL, url:str = OLLAMA_URL):
        self.model = model
        self.url = url

    def clean(self, text: str) -> str:
        """Удаляем <think>, ``` и английскую преамбулу."""
        # убираем теги <think>
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
        # убираем markdown‑блоки
        text = text.replace("```", "").strip()
        # убираем leading english до первой русской буквы
        m = re.search(r"[А-Яа-яЁё]", text)
        return text[m.start():] if m else text

    def _chat(self,
            messages: List[Dict[str, str]],
            options: Dict[str, Any]):
        
        try:
            return ollama.chat(self.model, messages=messages, options=options)
        except Exception as e:
            raise RuntimeError(f"Generation error: {e}")
        
    def generate_response(self,
                          transcript:str,
                          opts: Dict[str, Any] | None = None,
                          system_prompt: str | None = None,
                          ):
        
        messages = [
        {"role": "system", "content": system_prompt},
        FEWSHOT_SUMMARY,
        {"role": "user", "content": transcript},
    ]
        resp = self._chat(messages, opts)
        result = self.clean(resp["message"]["content"])
        return {"result": result}
    
    def generate_text_sim(self,
                          transcript:str,
                          opts: Dict[str, Any] | None = None,
                          system_prompt: str | None = None,
                          ):
        
        messages = [
        {"role": "system", "content": system_prompt},
        FEWSHOT_SUMMARY,
        {"role": "user", "content": transcript},
    ]
        resp = self._chat(messages, opts)
        result = self.clean(resp["message"]["content"])
        return {"result": result}
