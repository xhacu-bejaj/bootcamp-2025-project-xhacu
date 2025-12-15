import time
from typing import Optional

from app.core.logging import log_api_call
from app.models.schemas import LLMOutput, LLMParams
from app.services.llm_client import LLMClient


class MockLLM(LLMClient):
    @log_api_call
    async def generate(self, prompt: str, params: Optional[LLMParams] = None) -> LLMOutput | None:
        start_time = time.perf_counter()

        mock_output = (
            f"[MOCK OUTPUT]\n{prompt}\n\n--- Mock processed output ---"
        )
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)

        # Start with default model info
        model_info = LLMParams(model="mock", temperature=0.5)

        # If custom params are provided, update the model_info
        if params:
            if params.model:
                model_info.model = params.model
            if params.temperature is not None:
                model_info.temperature = params.temperature

        return LLMOutput(
            output_text=mock_output,
            model_info=model_info,
            latency_ms=latency_ms,
        )
