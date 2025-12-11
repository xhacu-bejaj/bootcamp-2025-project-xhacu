import time

from app.core.logging import log_api_call
from app.models.schemas import PredictResponse, ModelInfo
from app.services.llm_client import LLMClient


class MockLLM(LLMClient):
    @log_api_call
    def generate(self, prompt: str, **kwargs) -> PredictResponse | None:
        start_time = time.perf_counter()

        prompt_id = kwargs.pop("prompt_id", "mock_id")
        prompt_version = kwargs.pop("prompt_version", 0)

        mock_output = (
            f"[MOCK OUTPUT]\n{prompt}\n\n--- Mock processed output ---"
        )
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)

        return PredictResponse(
            output_text=mock_output,
            model_info=ModelInfo(model="mock", temperature=0.5),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            latency_ms=latency_ms,
        )
