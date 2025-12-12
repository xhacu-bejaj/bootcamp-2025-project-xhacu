from dataclasses import dataclass
import time
from typing import Optional

from openai import AsyncOpenAI

from app.services.llm_client import LLMClient
from app.models.schemas import LLMParams, OutputSchema, LLMOutput
from app.core.config import settings
from app.core.exceptions import ConfigurationError, LLMGenerationError
from app.core.logging import setup_logging, log_api_call


setup_logging()

@dataclass
class OpenaiLLM(LLMClient):
    model: str = "gpt-4o-mini"
    temperature: float = 0.5

    @log_api_call
    def __post_init__(self):
        OPENAI_API_KEY = settings.OPENAI_API_KEY
        if not OPENAI_API_KEY:
            raise ConfigurationError("OPENAI_API_KEY is missing or empty.")

        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    @log_api_call
    async def generate(self, prompt: str, params: Optional[LLMParams] = None) -> LLMOutput | None:
        # Determine actual temperature to use
        actual_temperature = params.temperature if params and params.temperature is not None else self.temperature

        json_instruction = (
            "Your sole output must be a valid JSON object. "
            "Strictly adhere to the structure with only an 'output_text' key. "
            "Place the result of the task in the **'output_text'** key. "
            "Do not include any other keys, explanations, or text outside the JSON block."
        )

        system_prompt = f"{prompt}\n\n{json_instruction}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": actual_temperature,
            "response_format": {"type": "json_object"},
        }

        start_time = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**api_params)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
        except Exception as e:
            raise LLMGenerationError(f"OpenAI API failed: {e}")

        json_string = response.choices[0].message.content

        if json_string is None:
            raise LLMGenerationError("OpenAI model returned no content.")

        try:
            llm_output = OutputSchema.model_validate_json(json_string)
        except Exception as e:
            raise LLMGenerationError(
                f"LLM failed to return valid JSON conforming to OutputSchema: {e}. Raw Text: {json_string}"
            )

        return LLMOutput(
            output_text=llm_output.output_text,
            model_info=LLMParams(model=self.model, temperature=actual_temperature),
            latency_ms=latency_ms,
        )
