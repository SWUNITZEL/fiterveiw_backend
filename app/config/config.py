from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MONGO_DB_NAME: str
    MONGO_DB_URL: str

settings = Settings()
