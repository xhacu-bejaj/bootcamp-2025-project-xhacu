import time
from typing import Optional

from app.core.logging import log_api_call
from app.models.schemas import LLMOutput, LLMParams
from app.services.llm_client import LLMClient


class MockLLM(LLMClient):
    @log_api_call
    def generate(self, prompt: str, params: Optional[LLMParams] = None) -> LLMOutput | None:
        start_time = time.perf_counter()

        mock_output = (
            f"[MOCK OUTPUT]\n{prompt}\n\n--- Mock processed output ---"
        )
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)

        return LLMOutput(
            output_text=mock_output,
            model_info=LLMParams(model="mock", temperature=0.5),
            latency_ms=latency_ms,
        )
