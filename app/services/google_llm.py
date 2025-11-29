from dataclasses import dataclass
import logging
from typing import Any
import json


from google import genai
from google.genai.types import GenerateContentConfig

from app.models.domain import Prompt
from app.services.llm_client import LLMClient
from app.core import config
from app.models.schemas import PredictResponse


global_settings = config.Settings()
#google_logger = logging.getLogger("GOOGLE")

@dataclass
class GoogleLLM(LLMClient):
    model: str='gemini-2.5-flash'
    temperature: float=0.5
    max_output_tokens: int=2000
    #SYSTEM_GUARDRAIL: str = "You are a helpful, ethical, and safe assistant. You must refuse requests that promote illegal acts, hate speech, or explicit content. Respond only to appropriate topics."


    def __post_init__(self):
        try:
            GOOGLE_API_KEY = global_settings.GOOGLE_API_KEY
        except Exception as e:
            #google_logger.error("GOOGLE_API_KEY is not set in the environment")
            raise ValueError("GOOGLE_API_KEY is not set in the environment")
        
        if not GOOGLE_API_KEY:
            #google_logger.error("GOOGLE_API_KEY is missing or empty.")
            raise ValueError("GOOGLE_API_KEY is missing or empty.")
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.config = GenerateContentConfig()
        #google_logger.info(f"GoogleAIClient initialized with model: {self.model}")

    def generate(self, template: str | None, document_text: str, **kwargs) -> PredictResponse | None:
        #google_logger.info(f"Generating content using Google client for prompt: '{prompt}...'")
        
        config_params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "system_instruction": template, 
            "response_mime_type":"application/json",
            "response_schema": PredictResponse
        }
        
        config_params.update(kwargs)
        config = GenerateContentConfig(**config_params)
        #####################################################
        ########## GEMINI INITIALIZED AND CONFIG SET ########
        
        response = self.client.models.generate_content( # fails here
            model=self.model, 
            contents=document_text, 
            config=config
        )
            
        if response.text:
            json_data = json.loads(response.text)
            model_info_dict = json_data.pop('model_info')
            model_info = dict(**model_info_dict)
            resp = PredictResponse(model_info=model_info, **json_data)
            return resp


    

        
            
            