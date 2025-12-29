from abc import ABC, abstractmethod

import polars as pl


class BaseTransformation(ABC):
    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        pass
