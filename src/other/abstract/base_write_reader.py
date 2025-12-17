from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

class BaseWriterReader(ABC):
    def __init__(self, file_path: Path):
        self.file_path = file_path

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
    @abstractmethod
    def read(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def write(self, df: pl.DataFrame, file_path: Path):
        pass