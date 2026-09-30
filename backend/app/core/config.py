from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union
from pathlib import Path
from pydantic import field_validator
import json


class Settings(BaseSettings):
    PROJECT_NAME: str = "Potato Disease Research Demo"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 7860  # Default 7860 for Hugging Face Spaces, 8000 for local development
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"
    CONFIGS_DIR: Path = BASE_DIR / "configs"

    # Default device configuration
    DEVICE: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            return json.loads(v)
        return v


settings = Settings()
