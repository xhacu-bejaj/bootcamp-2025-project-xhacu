from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    FILE_SNAPSHOT: bool = False

    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str = ""
    ASYNC_DB: str = ""
    MONGODB_URI: str = ""
    # Accept alternate connection string name used in .env
    MONGODB_CONNECTION_STRING: str = ""
    # Database and collection names (defaults match your setup)
    MONGODB_DB_NAME: str = "data"
    MONGODB_PROMPTS_COLLECTION: str = "prompts_db"
    MONGODB_LOGS_COLLECTION: str = "logs"

    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    LOG_FILE_PATH: str = "service-history.log"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
# Normalize connection string fallback: prefer explicit MONGODB_URI, else MONGODB_CONNECTION_STRING
if not settings.MONGODB_URI and settings.MONGODB_CONNECTION_STRING:
    settings.MONGODB_URI = settings.MONGODB_CONNECTION_STRING

global_settings = settings
