import time

from app.core.logging import log_api_call
from app.models.domain import Prompt
from app.models.schemas import PredictResponse, ModelInfo
from app.services.llm_client import LLMClient



class MockLLM(LLMClient):
    @log_api_call
    def generate(self, active_prompt: Prompt, document_text: str, **params):
        start_time = time.perf_counter()
        mock_output = f"[MOCK OUTPUT]\n{active_prompt.template}\n\nDocument: {document_text}\n\n--- Mock processed output ---"
        end_time = time.perf_counter()
        latency_ms = int((end_time - start_time) * 1000)

        return PredictResponse(
            output_text=mock_output,
            model_info=ModelInfo(model="mock", temperature=0.5),
            prompt_id=active_prompt.id,
            prompt_version=active_prompt.version,
            latency_ms=latency_ms,
        )
