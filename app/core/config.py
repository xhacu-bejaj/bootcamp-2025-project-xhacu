from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    FILE_SNAPSHOT: bool = False

    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "service-history.log"

    DATABASE_URL: str = ""
    ASYNC_DB: str = ""
    
    MONGODB_URI: str = ""
    MONGODB_CONNECTION_STRING: str = ""
 
    MONGODB_DB_NAME: str = "data"
    MONGODB_PROMPTS_COLLECTION: str = "prompts_db"
    MONGODB_LOGS_COLLECTION: str = "logs"
    #MONGODB_ACTIVE_COLLECTION: str = "active_prompts"
    
    CHROMA_PERSIST_DIR: str = "./var/chroma_db"
    CHROMA_COLLECTION_NAME: str = "chunks"
    MAX_CHUNK_LENGTH: int = 1000

    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
# Normalize connection string fallback: prefer explicit MONGODB_URI, else MONGODB_CONNECTION_STRING
if not settings.MONGODB_URI and settings.MONGODB_CONNECTION_STRING:
    settings.MONGODB_URI = settings.MONGODB_CONNECTION_STRING
