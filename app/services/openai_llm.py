from dataclasses import dataclass
import time
from typing import Set

from fastapi import HTTPException
from openai import OpenAI

from app.models.domain import Prompt
from app.services.llm_client import LLMClient
from app.models.schemas import ModelInfo, OpenaiOutputSchema, PredictResponse
from app.core.config import settings
from app.core.logging import setup_logging, log_api_call


setup_logging()

@dataclass
class OpenaiLLM(LLMClient):
    model: str = "gpt-4o-mini"
    temperature: float = 0.5
    provider_name: str = "openai"

    @log_api_call
    def __post_init__(self):
        try:
            OPENAI_API_KEY = settings.OPENAI_API_KEY
        except Exception:
            raise ValueError("OPENAI_API_KEY is not set in the environment")

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing or empty.")

        self.client = OpenAI(api_key=OPENAI_API_KEY)

    @log_api_call
    def generate(
        self, active_prompt: Prompt, document_text: str, **kwargs
    ) -> PredictResponse | None:
        VALID_CONFIG_KEYS: Set[str] = {
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
        }

        final_temperature: float = kwargs.get("temperature", self.temperature)

        json_instruction = (
            "Your sole output must be a valid JSON object. "
            "Strictly adhere to the structure of the PredictResponse schema. "
            "Place the result of the task (the translation) in the **'output_text'** key. "
            "Do not include any other keys, explanations, or text outside the JSON block."
        )

        # Render prompt template with Jinja2
        rendered_prompt = active_prompt.render({
            "document_text": document_text,
            "json_instruction": json_instruction
        })

        messages = [
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": document_text},
        ]

        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": final_temperature,
            "response_format": {"type": "json_object"},
        }

        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in VALID_CONFIG_KEYS and k not in api_params
        }
        api_params.update(filtered_kwargs)

        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**api_params)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OpenAI API failed: {e}")

        json_string = response.choices[0].message.content

        if json_string is None:
            raise ValueError("OpenAI model returned no content.")

        try:
            llm_output = OpenaiOutputSchema.model_validate_json(json_string)
        except Exception as e:
            raise ValueError(
                f"LLM failed to return valid JSON conforming to PredictResponse schema: {e}. Raw Text: {json_string}"
            )

        predicted_response = PredictResponse(
            output_text=llm_output.output_text,
            model_info=ModelInfo(model=self.model, temperature=final_temperature),
            prompt_id=active_prompt.id,
            prompt_version=active_prompt.version,
            latency_ms=latency_ms,
        )
        return predicted_response
