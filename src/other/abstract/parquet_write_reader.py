from pathlib import Path
from typing import Optional

import polars as pl

from src.other.abstract.base_write_reader import BaseWriterReader


class ParquetWriterReader(BaseWriterReader):
    """Writer/Reader implementation for Parquet files."""
    
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self._df: Optional[pl.DataFrame] = None

    def read(self) -> pl.DataFrame:
        """Read data from the parquet file."""
        self._validate_file_exists()
        self._df = pl.read_parquet(self.file_path)
        return self._df
    
    def write(self, df: pl.DataFrame, file_path: Optional[Path] = None) -> None:
        """Write data to a parquet file. Uses original path if no path specified."""
        output_path = file_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
