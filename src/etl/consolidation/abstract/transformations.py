from typing import List, Any
from src.etl.consolidation.abstract.base_transformation import BaseTransformation

import polars as pl


class TransformationsManager:
    """Manages and executes a pipeline of transformations on a DataFrame."""
    
    def __init__(self, transformations: List[BaseTransformation]):
        self.transformations = transformations

    def run_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute all transformations in sequence."""
        for transformation in self.transformations:
            df = transformation.transform(df)
        return df


# ============================================================================
# FILTER TRANSFORMATIONS - Remove rows or filter elements
# ============================================================================

class Filter_RowsNullEmpty(BaseTransformation):
    """Filter out rows where a column is null or empty (for list columns)."""
    
    def __init__(self, col: str, is_list: bool = False):
        self.col = col
        self.is_list = is_list

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.is_list:
            return df.filter(
                pl.col(self.col).is_not_null() & 
                (pl.col(self.col).list.len() > 0)
            )
        return df.filter(pl.col(self.col).is_not_null())


class Filter_ListElements(BaseTransformation):
    """Filter list elements that appear less than threshold times across the dataset."""
    
    def __init__(self, col: str, threshold: int):
        self.col = col
        self.threshold = threshold

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        valid = (
            df.explode(self.col)
            .group_by(self.col)
            .len()
            .filter(pl.col("len") >= self.threshold)
            .select(self.col)
        )

        df_with_idx = df.with_row_index("__row_id")
        other_cols = [c for c in df.columns if c != self.col]
        
        df_clean = (
            df_with_idx
            .explode(self.col)        
            .join(valid, on=self.col, how="inner")
            .group_by("__row_id")       
            .agg([pl.col(self.col)] + [pl.col(c).first() for c in other_cols])
            .drop("__row_id")
        )
        
        return df_clean


# ============================================================================
# CREATE TRANSFORMATIONS - Produce new columns
# ============================================================================

class Create_ConstantColumn(BaseTransformation):
    """Add a new column with a constant value."""
    
    def __init__(self, col: str, value: Any):
        self.col = col
        self.value = value

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit(self.value).alias(self.col))


class Create_PlayerCountColumns(BaseTransformation):
    """One-hot encode player counts from min/max player columns."""
    
    def __init__(self, min_col: str, max_col: str, max_cap: int):
        self.min_col = min_col
        self.max_col = max_col
        self.max_cap = max_cap

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        expressions = [
            pl.when((pl.col(self.min_col) <= i) & (pl.col(self.max_col) >= i))
            .then(1)
            .otherwise(0)
            .alias(f"players_{i}")
            for i in range(1, self.max_cap + 1)
        ]
        return df.with_columns(expressions).drop([self.min_col, self.max_col])


class Create_PopularityScore(BaseTransformation):
    """Create a normalized popularity score from owned/wanted/wished columns."""
    
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

        df = df.with_columns(raw_expr.alias("__raw_popularity"))

        condition = pl.col(self.filter_column) == self.value

        subset_min = pl.col("__raw_popularity").filter(condition).min()
        subset_max = pl.col("__raw_popularity").filter(condition).max()
        
        norm_calc = (
            pl.when(subset_min == subset_max)
            .then(0.0)
            .otherwise((pl.col("__raw_popularity") - subset_min) / (subset_max - subset_min))
        )

        df = df.with_columns(
            pl.when(condition)
            .then(norm_calc)
            .otherwise(None) 
            .alias("popularity_score")
        )
        
        cols_to_drop = ["__raw_popularity", self.owned_col, self.wanted_col, self.wished_col]
        return df.drop(cols_to_drop)


class Create_OneHotFromList(BaseTransformation):
    """One-hot encodes a column containing lists of strings."""
    
    def __init__(self, col: str):
        self.col = col

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        unique_vals = (
            df.select(pl.col(self.col).explode().unique().drop_nulls())
            .get_column(self.col)
            .sort()
            .to_list()
        )

        expressions = [
            pl.col(self.col)
            .list.contains(val)
            .cast(pl.Int64)
            .alias(f"{self.col}_{val}") 
            for val in unique_vals
        ]

        df = df.with_columns(expressions)
        return df.drop(self.col)


# ============================================================================
# TRANSFORM TRANSFORMATIONS - Modify existing columns in-place
# ============================================================================

class Transform_ColumnTypes(BaseTransformation):
    """Cast columns to specified data types."""
    
    TYPE_MAP = {
        "int": pl.Int64,
        "float": pl.Float64,
        "str": pl.Utf8,
        "bool": pl.Boolean,
    }
    
    def __init__(self, columns: dict[str, str]):
        """
        Args:
            columns: Dict mapping column names to target types.
                     Types: 'int', 'float', 'str', 'bool'
        """
        self.columns = columns

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cast_exprs = []
        for col_name, type_name in self.columns.items():
            if type_name not in self.TYPE_MAP:
                raise ValueError(f"Unknown type '{type_name}'. Valid types: {list(self.TYPE_MAP.keys())}")
            target_type = self.TYPE_MAP[type_name]
            cast_exprs.append(pl.col(col_name).cast(target_type, strict=False).alias(col_name))
        
        return df.with_columns(cast_exprs)


class Transform_NormalizeColumn(BaseTransformation):
    """Normalize a column to 0-1 range for rows matching a filter condition."""
    
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


# ============================================================================
# UPDATE TRANSFORMATIONS - Conditionally modify values in existing columns
# ============================================================================

class Update_ColumnByThreshold(BaseTransformation):
    """Set a filter column value based on whether a target column meets a threshold."""
    
    def __init__(self, col: str, threshold: float, filter_column: str, value: Any):
        self.col = col
        self.threshold = threshold
        self.filter_column = filter_column
        self.value = value

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.col(self.col) >= self.threshold)
            .then(pl.lit(self.value))
            .otherwise(pl.col(self.filter_column)) 
            .alias(self.filter_column)
        )