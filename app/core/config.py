from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    FILE_SNAPSHOT: bool = False
    LOG_LEVEL: str = "INFO"
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()