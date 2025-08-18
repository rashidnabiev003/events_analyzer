import os
from pydantic import BaseModel
from typing import List, Dict, Any, Sequence, Optional
from src.schemas.main_schemas import ChatItem, EngineConfig, RiskResp
import json
import re
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional jsonschema validation
try:
	from jsonschema import validate  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without jsonschema
	def validate(*_args: Any, **_kwargs: Any) -> None:
		return

# Optional vLLM import with minimal fallbacks for tests
try:
	from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover - provide light stubs for testing
	class SamplingParams:  # minimal interface used in code
		def __init__(
			self,
			temperature: float = 0.3,
			top_p: float = 0.92,
			top_k: int = 50,
			max_tokens: int = 2000,
			repetition_penalty: float = 1.2,
			seed: int = 42,
		):
			self.temperature = temperature
			self.top_p = top_p
			self.top_k = top_k
			self.max_tokens = max_tokens
			self.repetition_penalty = repetition_penalty
			self.seed = seed

	class LLM:  # pragma: no cover - placeholder; tests monkeypatch this
		def __init__(self, *args: Any, **kwargs: Any) -> None:
			pass
		def chat(self, prompts: Sequence[Sequence[Dict[str, str]]], sampling_params: SamplingParams):
			raise RuntimeError("LLM backend is not available. Tests should monkeypatch LLM.")


def extract_json(text: str) -> dict:
	# Убираем лишние теги и извлекаем JSON, возвращая {} при ошибках
	try:
		text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
		text = re.sub(r"```json\s*|\s*```", "", text, flags=re.I)
		m = re.search(r"\{[\s\S]*\}", text)
		if not m:
			return {}
		return json.loads(m.group(0))
	except Exception:
		return {}


class VLLMEngine:
	def __init__(self, engine_config: EngineConfig, system_prompt: str | None = None):

		self.vllm_engine = LLM(
			model=engine_config.model_path,
			trust_remote_code=engine_config.trust_remote_code,
			gpu_memory_utilization=engine_config.gpu_memory_utilization,
			quantization=engine_config.quantization,
			tensor_parallel_size=engine_config.tensor_parallel_size,
		)
		self.system_prompt = system_prompt
		self.batch_size = engine_config.max_batch_size
		self.base_seed = 42

	def chat_batch(
		self,
		items: Optional[Sequence[ChatItem]] = None,
		sampling_params: Dict[str, Any] | None = None,
		json_schema: Any = None,
		max_retries: int = 1,
	) -> List[Dict[str, Any]]:
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
					{'role': 'system', 'content': item.system_prompt},
					{'role': 'user', 'content': item.prompt},
				]
				for item in items
			]

		responses = [None] * len(full_prompts)
		remaining = list(range(len(full_prompts)))

		for attempt in range(max_retries + 1):
			if not remaining:
				break
			sampling_params.seed = self.base_seed + attempt

			for i in range(0, len(remaining), self.batch_size):
				idx_batch = remaining[i : i + self.batch_size]
				prompts_batch = [full_prompts[j] for j in idx_batch]
				try:
					outputs = self.vllm_engine.chat(prompts_batch, sampling_params)  # type: ignore[attr-defined]
				except Exception as e:
					print(f"[vllm] Ошибка батча attempt={attempt} idx={idx_batch}: {e}")
					continue

				for out_idx, out in enumerate(outputs):
					orig_idx = idx_batch[out_idx]
					text = out.outputs[0].text.strip() if out.outputs else ""
					try:
						data = extract_json(text)
						if json_schema is not None and hasattr(json_schema, 'model_json_schema'):
							validate(instance=data, schema=json_schema.model_json_schema())
						responses[orig_idx] = data
						remaining.remove(orig_idx)
					except Exception:
						# Всё ещё невалиден — оставляем в remaining
						pass

		# 2) Для каждого оставшегося — индивидуальные попытки до успеха
		for idx in remaining:
			attempt = 0
			while True:
				seed = self.base_seed + max_retries + attempt + 1
				sampling_params.seed = seed
				try:
					output = self.vllm_engine.chat([full_prompts[idx]], sampling_params)[0]  # type: ignore[attr-defined]
					text = output.outputs[0].text.strip() if output.outputs else ""
					data = extract_json(text)
					if json_schema is not None and hasattr(json_schema, 'model_json_schema'):
						validate(instance=data, schema=json_schema.model_json_schema())
					responses[idx] = data
					break
				except Exception as e:
					print(f"[vllm][fallback] idx={idx}, attempt={attempt}: {e}")
					attempt += 1

		return responses
