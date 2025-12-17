from pathlib import Path

import polars as pl

from other.abstract.base_write_reader import BaseWriteReader

class ParquetWriteReader(BaseWriteReader):
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self.df = None

    def read(self) -> pl.Dataframe:
        self.df = pl.read_parquet(self.file_path)
        return self.df
    
    def write(self, df: pl.DataFrame, file_path: Path):
        df.write_parquet(file_path)
