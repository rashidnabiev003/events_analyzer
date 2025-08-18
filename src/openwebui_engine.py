# src/openwebui_engine.py
import os
import json
import re
from typing import List, Dict, Any, Sequence, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import requests
# подправьте импорт под ваш layout (в model.py используется "src.schemas...")
from src.schemas.main_schemas import ChatItem

# опциональная валидация jsonschema
try:
    from jsonschema import validate  # type: ignore
except Exception:
    def validate(*_args: Any, **_kwargs: Any) -> None:
        return

def extract_json(text: str) -> dict:
    # убираем "<think>" и Markdown-блоки кода
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.I)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("JSON not found in model output")
    return json.loads(m.group(0))

class OpenWebUIEngine:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_concurrency: int = 8,
        system_prompt: Optional[str] = None,
    ):
        base = (base_url or os.getenv("OPENWEBUI_BASE_URL"))
        self.completions_url = f"{base}/api/chat/completions"  # см. оф. доки
        self.models_url = f"{base}/api/models"
        self.api_key = api_key or os.getenv("OPENWEBUI_API_KEY")
        self.model = model or os.getenv("OPENWEBUI_MODEL")
        self.timeout = float(timeout)
        self.max_concurrency = max(1, int(max_concurrency))
        self.system_prompt_default = system_prompt

        self._headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }

    def _single(
        self,
        item: ChatItem,
        json_schema: Any,
        max_retries: int,
    ) -> Dict[str, Any]:
        # если у ChatItem нет system_prompt — используем дефолт, если задан
        sys_msg = item.system_prompt or self.system_prompt_default
        msgs = []
        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})
        msgs.append({"role": "user", "content": item.prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
        }

        last_exc: Optional[Exception] = None
        for _ in range(max_retries + 1):
            try:
                r = requests.post(
                    self.completions_url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                # специфика OpenAI-совместимых ответов
                text = data["choices"][0]["message"]["content"]
                obj = extract_json(text)
                if json_schema is not None and hasattr(json_schema, "model_json_schema"):
                    validate(instance=obj, schema=json_schema.model_json_schema())
                return obj
            except Exception as e:
                last_exc = e
                continue

        return {"risk": 0.0, "reason": f"openwebui_error: {last_exc}"}

    def chat_batch(
        self,
        items: Optional[Sequence[ChatItem]] = None,
        sampling_params: Dict[str, Any] | None = None,  # совместимость сигнатуры
        json_schema: Any = None,
        max_retries: int = 1,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []
        results: List[Optional[Dict[str, Any]]] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futs = {
                ex.submit(self._single, it, json_schema, max_retries): i
                for i, it in enumerate(items)
            }
            # прогресс по завершению фьюч
            with tqdm(total=len(futs), desc="OpenWebUI batch", unit="req") as pbar:
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        results[i] = fut.result()
                    except Exception as e:
                        results[i] = {"risk": 0.0, "reason": f"openwebui_exc: {e}"}
                    finally:
                        pbar.update(1)

        return [r or {"risk": 0.0, "reason": ""} for r in results]
