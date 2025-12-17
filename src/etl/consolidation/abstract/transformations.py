from typing import List, Any
from src.etl.consolidation.abstract.base_transformation import BaseTransformation

import polars as pl

class TransformationsManager:
    def __init__(self, transformations: List[BaseTransformation]):
        self.transformations = transformations

    def run_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        for tranformation in self.transformations:
            df = tranformation.transform(df)

        return df

class emptynull_Filter(BaseTransformation):
    def __init__(self, col: str):
        self.col = col

    def transform(self, df: pl.Dataframe, ) -> List[str]:
        df_filtered = df.filter(
            pl.col(self.col).is_not_null() & 
            pl.col(self.col).list.len() > 0
        )

        return df_filtered

class threshold_column_Filter(BaseTransformation):
    def __init__(self, target_col: str, threshold: float, filter_column: str, value: Any):
        self.target_col = target_col
        self.threshold = threshold
        self.filter_column = filter_column
        self.value = value

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.col(self.target_col) >= self.threshold)
            .then(pl.lit(self.value))
            .otherwise(pl.col(self.filter_column)) 
            .alias(self.filter_column)
        )
    
class threshold_listelement_Filter(BaseTransformation):
    def __init__(self, col: str, threshold: int):
        self.col = col
        self.threshold = threshold

    def transform(self, df: pl.DataFrame) -> List[str]:
        valid = (
            df.explode(self.col)
            .group_by(self.col)
            .len()
            .filter(pl.col("len") >= self.threshold)
            .select(self.col)
        )

        df_clean = (
            df.with_row_index("row_id")
            .explode(self.col)        
            .join(
                valid, 
                on=self.col, 
                how="inner"           
            )
            .group_by("row_id")       
            .agg(pl.col(self.col))         
            .drop("row_id")          
        )
        
        return df_clean

class playercounts_Encode(BaseTransformation):
    def __init__(self, min_col: str, max_col: str, max_cap: int):
        self.min_col = min_col
        self.max_col = max_col
        self.max_cap = max_cap

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        expressions = []
        for i in range(1, self.max_cap + 1):
            expr = (
                pl.when((pl.col(self.min_col) <= i) & (pl.col(self.max_col) >= i))
                .then(1)
                .otherwise(0)
                .alias(f"players_{i}")
            )
            expressions.append(expr)

        return df.with_columns(expressions).drop([self.min_col, self.max_col])


import polars as pl
from typing import Any

class popularityscore_Create(BaseTransformation):
    def __init__(self, owned_col: str, wanted_col: str, wished_col: str, 
                 filter_column: str, value: Any,
                 weights: tuple = (1.0, 2.0, 0.5)):
        
        self.owned_col = owned_col
        self.wanted_col = wanted_col
        self.wished_col = wished_col
        self.filter_column = filter_column
        self.value = value
        self.weights = weights

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        w_owned, w_wanted, w_wished = self.weights

        raw_expr = (
            (pl.col(self.owned_col) * w_owned) +
            (pl.col(self.wanted_col) * w_wanted) +
            (pl.col(self.wished_col) * w_wished)
        )

        df = df.with_columns(raw_expr.alias("raw_popularity"))

        condition = pl.col(self.filter_column) == self.value

        subset_min = pl.col("raw_popularity").filter(condition).min()
        subset_max = pl.col("raw_popularity").filter(condition).max()
        
        norm_calc = (
            pl.when(subset_min == subset_max)
            .then(0.0)
            .otherwise((pl.col("raw_popularity") - subset_min) / (subset_max - subset_min))
        )

        df = df.with_columns(
            pl.when(condition)
            .then(norm_calc)
            .otherwise(None) 
            .alias("popularity_score")
        )
        cols_to_drop = ["raw_popularity"]
        cols_to_drop.extend([self.owned_col, self.wanted_col, self.wished_col])
        
        return df.drop(cols_to_drop)

class normalizecolumn_Transformation(BaseTransformation):
    def __init__(self, col: str, filter_column: str, value: Any):
        self.col = col
        self.filter_column = filter_column
        self.value = value

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        condition = pl.col(self.filter_column) == self.value

        subset_min = pl.col(self.col).filter(condition).min()
        subset_max = pl.col(self.col).filter(condition).max()
        
        norm_calc = (
            pl.when(subset_min == subset_max)
            .then(0.0)
            .otherwise((pl.col(self.col) - subset_min) / (subset_max - subset_min))
        )

        return df.with_columns(
            pl.when(condition)
            .then(norm_calc)
            .otherwise(pl.col(self.col))
            .alias(self.col)
        )
    
class newcolumn_value_Add(BaseTransformation):
    def __init__(self, column_name: str, value: Any):
        self.column_name = column_name
        self.value = value

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.lit(self.value).alias(self.column_name)
        )