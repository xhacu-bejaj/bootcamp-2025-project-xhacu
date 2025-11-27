from dataclasses import dataclass
import logging
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.services.llm_client import LLMClient
from app.models.schemas import PredictRequest, PredictResponse
from app.core import config

global_settings = config.Settings()
#openai_logger = logging.getLogger("OPENAI")

@dataclass
class OpenaiLLM(LLMClient):
  model: str='gpt-4o-mini'
  temperature: float=0.5
  max_tokens: int=2000

  def __post_init__(self):
      try:
        OPENAI_API_KEY = global_settings.OPENAI_API_KEY
      except Exception as e:
            #openai_logger.error("OPENAI_API_KEY is not set in the environment")
            raise ValueError("OPENAI_API_KEY is not set in the environment")
      
      if not OPENAI_API_KEY:
            #openai_logger.error("OPENAI_API_KEY is missing or empty")
            raise ValueError("OPENAI_API_KEY is missing or empty")
      
      self.client = OpenAI(api_key=OPENAI_API_KEY)
      #openai_logger.info(f"OpenAIClient initialized with model: {self.model}")

  def generate(self, prompt: str, **kwargs) -> GenerateResponse: 
    #openai_logger.info(f"Generating content using OpenAI client for prompt: '{prompt}...'")
      
    call_params = {
        "model": self.model,
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        **kwargs
      }

    messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful, ethical, and safe assistant. You must refuse requests that promote illegal acts, hate speech, or explicit content. Respond only to appropriate topics."},
            {"role": "user", "content": prompt}
        ]
    try:
        response = self.client.chat.completions.create(
            messages=messages,
            **call_params
        )
    except Exception as e:
        #openai_logger.error(f"OpenAI client failed to generate content: {e}")
        raise ValueError("Client 'Openai' failed to generate content") from e
          
    try:
        message= response.choices[0].message
        content = message.content
        
        if not content:
            #openai_logger.warning("OpenAI returned a message with no content (may be filtered).")
            return GenerateResponse(response=None) 
        
        #openai_logger.info("OpenAI client successfully generated content.")
        return GenerateResponse(response=content)
        
    except IndexError:
        #openai_logger.error("OpenAI response contained no choices/messages.")
        raise ValueError("Failed to get a valid response from the API: no choices.")
    except Exception as e:
        #openai_logger.error(f"Error processing OpenAI response: {e}")
        raise ValueError("Failed to process API response structure.") from e