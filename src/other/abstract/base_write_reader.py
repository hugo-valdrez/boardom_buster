from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import polars as pl


class BaseWriterReader(ABC):
    """Abstract base class for reading and writing data files."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def _validate_file_exists(self) -> None:
        """Validate that the file exists before reading."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
    @abstractmethod
    def read(self) -> pl.DataFrame:
        """Read data from the file path."""
        pass

    @abstractmethod
    def write(self, df: pl.DataFrame, file_path: Optional[Path] = None) -> None:
        """Write data to the specified file path or the original file path."""
        pass