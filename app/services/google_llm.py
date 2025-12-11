from dataclasses import dataclass
import time
from typing import Set

from fastapi import HTTPException
from google import genai
from google.genai.types import GenerateContentConfig

from app.services.llm_client import LLMClient
from app.core.config import settings
from app.models.schemas import ModelInfo, OutputSchema, PredictResponse
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
            raise ValueError("GOOGLE_API_KEY is missing or empty.")

        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    @log_api_call 
    def generate(self, prompt: str, **params) -> PredictResponse | None:
        VALID_CONFIG_KEYS: list[str] = [
            "temperature",
            "max_output_tokens",
            "top_k",
            "top_p",
        ]

        prompt_id = params.pop("prompt_id", None)
        prompt_version = params.pop("prompt_version", None)
        document_text = params.pop("document_text", None)
        if not all([prompt_id, prompt_version, document_text]):
            raise ValueError(
                "prompt_id, prompt_version, and document_text must be provided."
            )

        final_temperature: float = params.get("temperature", self.temperature)

        config_params = {
            "temperature": final_temperature,
            "response_mime_type": "application/json",
            "response_schema": OutputSchema,
        }

        filtered_params = {
            k: v
            for k, v in params.items()
            if k in VALID_CONFIG_KEYS and k not in config_params
        }
        config_params.update(filtered_params)

        try:
            config: GenerateContentConfig = GenerateContentConfig(**config_params)
        except Exception as e:
            raise ValueError(
                f"Failed to create GenerateContentConfig: {e}. Keys passed: {list(config_params.keys())}"
            )
        start_time = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=f"models/{self.model}",
                contents=prompt,
                config=config,
            )
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
        except Exception as e:
            raise ValueError(f"Model failed to generate content: {e}")

        json_string = response.text
        if json_string is not None:
            try:
                llm_output = OutputSchema.model_validate_json(json_string)
            except Exception as e:
                raise ValueError(
                    f"LLM failed to return valid JSON conforming to PredictResponse schema: {e}. Raw Text: {json_string}"
                )

        predicted_response = PredictResponse(
            output_text=llm_output.output_text,
            model_info=ModelInfo(model=self.model, temperature=final_temperature),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            latency_ms=latency_ms,
        )
        return predicted_response
