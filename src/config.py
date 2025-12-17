import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    PATHS: Dict[str, Path] = {}
    INGESTION: Dict[str, Any] = {}
    ETL: Dict[str, Any] = {}
    ML: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_yaml_config()

    def _load_yaml_config(self):
        yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {yaml_path}")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        self.PATHS = {
            "raw_data": PROJECT_ROOT / config['paths']['raw_data'],
            "processed_data": PROJECT_ROOT / config['paths']['processed_data'],
            "logs": PROJECT_ROOT / config['paths']['logs']
        }

        self.INGESTION = config.get('ingestion', {})

        self.ETL = config.get('etl', {})

        self.ML = config.get('ml', {})

settings = Settings()