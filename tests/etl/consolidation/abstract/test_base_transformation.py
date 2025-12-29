"""Tests for src/etl/consolidation/abstract/base_transformation.py"""
import pytest
import polars as pl

from src.etl.consolidation.abstract.base_transformation import BaseTransformation


class TestBaseTransformation:
    """Test suite for the BaseTransformation abstract class."""
    
    def test_requires_transform_method(self):
        """Test that subclasses must implement transform method."""
        class IncompleteTransformation(BaseTransformation):
            pass
        
        with pytest.raises(TypeError):
            IncompleteTransformation()
    
    def test_valid_subclass(self):
        """Test that a valid subclass can be instantiated."""
        class ValidTransformation(BaseTransformation):
            def transform(self, df: pl.DataFrame) -> pl.DataFrame:
                return df
        
        transformation = ValidTransformation()
        assert isinstance(transformation, BaseTransformation)
        
        df = pl.DataFrame({"col": [1, 2, 3]})
        result = transformation.transform(df)
        assert result.equals(df)
