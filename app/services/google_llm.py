from dataclasses import dataclass
import time
from typing import Optional

from google import genai
from google.genai.types import GenerateContentConfig

from app.models.schemas import LLMOutput, LLMParams, OutputSchema
from app.services.llm_client import LLMClient
from app.core.config import settings
from app.core.exceptions import ConfigurationError, LLMGenerationError
from app.core.logging import setup_logging, log_api_call


setup_logging()


@dataclass
class GoogleLLM(LLMClient):
    model: str = "gemini-2.5-flash"
    temperature: float = 0.5

    @log_api_call
    def __post_init__(self):
        GOOGLE_API_KEY = settings.GOOGLE_API_KEY
        if not GOOGLE_API_KEY:
            raise ConfigurationError("GOOGLE_API_KEY is missing or empty.")

        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    @log_api_call 
    async def generate(self, prompt: str, params:Optional[LLMParams] = None) -> LLMOutput | None:

        # Determine the actual temperature to use
        actual_temperature = params.temperature if params and params.temperature is not None else self.temperature
        
        config_params = {
            "temperature": actual_temperature,
            "response_mime_type": "application/json",
            "response_schema": OutputSchema.model_json_schema(), # Ensures output conforms to OutputSchema
        }

        try:
            config: GenerateContentConfig = GenerateContentConfig(**config_params)
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to create GenerateContentConfig: {e}. Keys passed: {list(config_params.keys())}"
            )
        start_time = time.perf_counter()
        try:
            response = await self.client.aio.models.generate_content(
                model=f"models/{self.model}",
                contents=prompt,
                config=config,
            )
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
        except Exception as e:
            raise LLMGenerationError(f"Model failed to generate content: {e}")

        json_string = response.text
        if json_string is not None:
            try:
                llm_output = OutputSchema.model_validate_json(json_string)
            except Exception as e:
                raise LLMGenerationError(
                    f"LLM failed to return valid JSON conforming to OutputSchema: {e}. Raw Text: {json_string}"
                )

        generated_content = LLMOutput(
            output_text=llm_output.output_text,
            model_info=LLMParams(model=self.model, temperature=actual_temperature),
            latency_ms=latency_ms,
        )
        return generated_content
