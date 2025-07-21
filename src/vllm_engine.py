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

    def chat_batch(self, items: Optional[Sequence[ChatItem]] = None,
                   sampling_params: Dict[str, Any] = None,
                   json_schema: Any = None,
                   max_retries: int = 2) -> List[Dict[str, Any]]:
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                max_tokens=600,
                repetition_penalty=1.35,
                seed=42,
            )

        full_prompts = [
                [
                    {'role':'system','content':item.system_prompt},
                    {'role':'user','content':item.prompt},
                ]
                for item in items
            ]
        
        responses = []
        for attempt in range(max_retries):
            outputs = self.vllm_engine.chat(full_prompts, sampling_params)

            current_responses = []
            for idx, out in enumerate(outputs):
                txt = out.outputs[0].text.strip() if out.outputs else ""
                try:
                    data = extract_json(txt)
                    if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
                        schema_dict = json_schema.model_json_schema()
                    else:
                        schema_dict = json_schema
                    validate(instance=data, schema=schema_dict)
                    current_responses.append(data)
                except ValidationError as e:
                    print(f"Validation failed for item {idx}, retrying: {e.message}")
                    current_responses.append(None)
                except Exception as e:
                    print(f"Parse error for item {idx}: {e}")
                    current_responses.append(None)

            # Обновляем full_prompts для неудачных запросов
            full_prompts = [
                prompt for prompt, resp in zip(full_prompts, current_responses) if resp is None
            ]
            if not full_prompts:
                break


        result = []
        for item in current_responses:
            if item:
                result.append(item)
            else:
                result.append({"risk": 0.0, "reason": "JSON validation failed"})
        
        return result