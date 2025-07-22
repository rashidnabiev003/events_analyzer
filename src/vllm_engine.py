import os
from pydantic import BaseModel
from typing import List, Dict, Any, Sequence, Optional
from schemas.main_schemas import ChatItem, EngineConfig, RiskResp
from vllm import LLM, SamplingParams
from jsonschema import validate, ValidationError
import json
import re 

def extract_json(text: str) -> dict:
    # Убираем лишние теги и извлекаем JSON
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    text = re.sub(r"```json\s*|\s*```", "", text, flags=re.I)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("JSON not found in model output")
    return json.loads(m.group(0))

class VLLMEngine:
    def __init__(self, engine_config: EngineConfig, system_prompt:str = None):

        self.vllm_engine = LLM(model=engine_config.model_path,
                               trust_remote_code=engine_config.trust_remote_code,
                               gpu_memory_utilization=engine_config.gpu_memory_utilization,
                               quantization=engine_config.quantization,
                               tensor_parallel_size=engine_config.tensor_parallel_size
                               )
        self.system_prompt = system_prompt 
        self.batch_size = engine_config.max_batch_size

    def chat_batch(self, items: Optional[Sequence[ChatItem]] = None,
                   sampling_params: Dict[str, Any] = None,
                   json_schema: Any = None,
                   max_retries: int = 1) -> List[Dict[str, Any]]:
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.92,
                top_k=50,
                max_tokens=2000,
                repetition_penalty=1.2,
                seed=42,
            )

        full_prompts = [
                [
                    {'role':'system','content':item.system_prompt},
                    {'role':'user','content':item.prompt},
                ]
                for item in items
            ]
        
        responses = [None] * len(full_prompts)
        remaining_indices = list(range(len(full_prompts)))

        for _ in range(max_retries):
            if not remaining_indices:
                break

            # Преобразуем json_schema в словарь
            if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
                schema_dict = json_schema.model_json_schema()
                if not isinstance(schema_dict, dict):
                    raise ValueError("model_json_schema() должен возвращать словарь")
            else:
                schema_dict = json_schema

            # Разбиваем на батчи
            for i in range(0, len(remaining_indices), self.batch_size):
                batch_indices = remaining_indices[i:i+self.batch_size]
                batch_prompts = [full_prompts[idx] for idx in batch_indices]

                try:
                    outputs = self.vllm_engine.chat(batch_prompts, sampling_params)
                    for idx, out in enumerate(outputs):
                        original_idx = batch_indices[idx]
                        txt = out.outputs[0].text.strip() if out.outputs else ""
                        data = extract_json(txt)
                        validate(instance=data, schema=schema_dict)
                        responses[original_idx] = data
                        remaining_indices.remove(original_idx)
                except ValidationError as e:
                    print(f"Validation failed for item {original_idx}: {e.message}")
                except Exception as e:
                    print(f"Parse error for item {original_idx}: {e}")

        # Заполняем пустые результаты заглушками
        for i in range(len(responses)):
            if responses[i] is None:
                responses[i] = {"risk": 0.0, "reason": "Validation failed"}
        return responses
