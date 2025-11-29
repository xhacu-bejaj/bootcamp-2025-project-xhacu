from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    FILE_SNAPSHOT: bool = False
    
    LOG_LEVEL: str = "INFO" 
    #DB_PATH: str = 'promptlog.db'
    GOOGLE_API_KEY: str = ''
    OPENAI_API_KEY: str = ''
    LOG_FILE_PATH: str = 'service-history.log'
    #MONGODB_CONNECTION_STRING: str = ''
    
    model_config = SettingsConfigDict(env_file=".env") 


settings = Settings()
