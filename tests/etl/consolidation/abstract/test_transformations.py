"""Tests for src/etl/consolidation/abstract/transformations.py"""

import polars as pl

from src.etl.consolidation.abstract.transformations import (
    Create_BGGLink,
    Create_ConstantColumn,
    Create_OneHotFromList,
    Create_PlayerCountColumns,
    Create_PopularityScore,
    Filter_ListElements,
    Filter_RowsNullEmpty,
    Transform_ClipValues,
    Transform_ColumnTypes,
    Transform_NormalizeColumn,
    TransformationsManager,
    Update_ColumnByThreshold,
)


class TestTransformationsManager:
    """Test suite for the TransformationsManager class."""

    def test_run_transformations_empty(self):
        """Test running with no transformations."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        manager = TransformationsManager(transformations=[])

        result = manager.run_transformations(df)

        assert result.equals(df)

    def test_run_transformations_single(self):
        """Test running a single transformation."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        transformation = Create_ConstantColumn(col="new_col", value=5)
        manager = TransformationsManager(transformations=[transformation])

        result = manager.run_transformations(df)

        assert "new_col" in result.columns
        assert result["new_col"].to_list() == [5, 5, 5]

    def test_run_transformations_multiple(self):
        """Test running multiple transformations in sequence."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        transformations = [
            Create_ConstantColumn(col="col1", value=10),
            Create_ConstantColumn(col="col2", value=20),
        ]
        manager = TransformationsManager(transformations=transformations)

        result = manager.run_transformations(df)

        assert "col1" in result.columns
        assert "col2" in result.columns
        assert result["col1"].to_list() == [10, 10, 10]
        assert result["col2"].to_list() == [20, 20, 20]


class TestFilter_RowsNullEmpty:
    """Test suite for Filter_RowsNullEmpty transformation."""

    def test_filter_null_values(self):
        """Test filtering rows with null values."""
        df = pl.DataFrame({"col": [1, None, 3]})
        transformation = Filter_RowsNullEmpty(col="col")

        result = transformation.transform(df)

        assert result.height == 2
        assert result["col"].to_list() == [1, 3]

    def test_filter_empty_lists(self):
        """Test filtering rows with empty lists."""
        df = pl.DataFrame({"col": [[1, 2], [], [3]]})
        transformation = Filter_RowsNullEmpty(col="col", is_list=True)

        result = transformation.transform(df)

        assert result.height == 2

    def test_filter_null_and_empty_lists(self):
        """Test filtering rows with null or empty lists."""
        df = pl.DataFrame({"col": [[1, 2], None, [], [3]]})
        transformation = Filter_RowsNullEmpty(col="col", is_list=True)

        result = transformation.transform(df)

        assert result.height == 2


class TestFilter_ListElements:
    """Test suite for Filter_ListElements transformation."""

    def test_filter_list_elements_basic(self):
        """Test filtering list elements below threshold."""
        df = pl.DataFrame({"items": [["A", "B"], ["B", "C"], ["A", "D"]]})
        transformation = Filter_ListElements(col="items", threshold=2)

        result = transformation.transform(df)

        # Only A and B appear >= 2 times
        assert result.height == 3
        # Check that rare elements are removed
        all_items = []
        for row in result["items"].to_list():
            all_items.extend(row)
        assert "D" not in all_items
        assert "C" not in all_items

    def test_filter_list_elements_removes_rows(self):
        """Test that rows with only rare elements are removed."""
        df = pl.DataFrame({"items": [["A"], ["A"], ["B"]], "id": [1, 2, 3]})
        transformation = Filter_ListElements(col="items", threshold=2)

        result = transformation.transform(df)

        # Only rows with "A" should remain
        assert result.height == 2
        assert result["id"].to_list() == [1, 2]


class TestCreate_ConstantColumn:
    """Test suite for Create_ConstantColumn transformation."""

    def test_create_constant_column_int(self):
        """Test creating a constant integer column."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        transformation = Create_ConstantColumn(col="new_col", value=42)

        result = transformation.transform(df)

        assert "new_col" in result.columns
        assert result["new_col"].to_list() == [42, 42, 42]

    def test_create_constant_column_string(self):
        """Test creating a constant string column."""
        df = pl.DataFrame({"col": [1, 2, 3]})
        transformation = Create_ConstantColumn(col="label", value="test")

        result = transformation.transform(df)

        assert result["label"].to_list() == ["test", "test", "test"]


class TestCreate_PlayerCountColumns:
    """Test suite for Create_PlayerCountColumns transformation."""

    def test_create_player_count_columns(self):
        """Test creating one-hot encoded player count columns."""
        df = pl.DataFrame({"min_players": [2, 1, 3], "max_players": [4, 2, 5]})
        transformation = Create_PlayerCountColumns(
            min_col="min_players", max_col="max_players", max_cap=5
        )

        result = transformation.transform(df)

        # Check that player columns are created
        assert "players_1" in result.columns
        assert "players_5" in result.columns

        # Check first row: supports 2-4 players
        assert result["players_1"][0] == 0
        assert result["players_2"][0] == 1
        assert result["players_3"][0] == 1
        assert result["players_4"][0] == 1
        assert result["players_5"][0] == 0

        # Check second row: supports 1-2 players
        assert result["players_1"][1] == 1
        assert result["players_2"][1] == 1
        assert result["players_3"][1] == 0
        assert result["players_4"][1] == 0
        assert result["players_5"][1] == 0

        # Check third row: supports 3-5 players
        assert result["players_1"][2] == 0
        assert result["players_2"][2] == 0
        assert result["players_3"][2] == 1
        assert result["players_4"][2] == 1
        assert result["players_5"][2] == 1

        # Check that original columns are removed
        assert "min_players" not in result.columns
        assert "max_players" not in result.columns


class TestCreate_PopularityScore:
    """Test suite for Create_PopularityScore transformation."""

    def test_create_popularity_score(self):
        """Test creating normalized popularity score."""
        df = pl.DataFrame(
            {
                "owned_by": [100, 200, 300],
                "wanted_by": [10, 20, 30],
                "wished_by": [5, 10, 15],
                "to_recommend": [1, 1, 1],
            }
        )
        transformation = Create_PopularityScore(
            owned_col="owned_by",
            wanted_col="wanted_by",
            wished_col="wished_by",
            filter_column="to_recommend",
            value=1,
            weights=(1.0, 2.0, 0.5),
        )

        result = transformation.transform(df)

        assert "popularity_score" in result.columns
        assert "owned_by" not in result.columns
        assert "wanted_by" not in result.columns
        assert "wished_by" not in result.columns

        # Check that scores are normalized to [0, 1]
        scores = result["popularity_score"].to_list()
        assert all(0 <= s <= 1 for s in scores if s is not None)

    def test_create_popularity_score_with_filter(self):
        """Test popularity score respects filter column."""
        df = pl.DataFrame(
            {
                "owned_by": [100, 200, 300],
                "wanted_by": [10, 20, 30],
                "wished_by": [5, 10, 15],
                "to_recommend": [1, 0, 1],
            }
        )
        transformation = Create_PopularityScore(
            owned_col="owned_by",
            wanted_col="wanted_by",
            wished_col="wished_by",
            filter_column="to_recommend",
            value=1,
            weights=(1.0, 2.0, 0.5),
        )

        result = transformation.transform(df)

        # Row with to_recommend=0 should have None for popularity_score
        assert result["popularity_score"][1] is None


class TestCreate_OneHotFromList:
    """Test suite for Create_OneHotFromList transformation."""

    def test_create_one_hot_from_list(self):
        """Test one-hot encoding of list column."""
        df = pl.DataFrame({"tags": [["A", "B"], ["B", "C"], ["A"]]})
        transformation = Create_OneHotFromList(col="tags")

        result = transformation.transform(df)

        assert "tags_A" in result.columns
        assert "tags_B" in result.columns
        assert "tags_C" in result.columns

        # Check first row: has A and B
        assert result["tags_A"][0] == 1
        assert result["tags_B"][0] == 1
        assert result["tags_C"][0] == 0

        # Check second row: has B and C
        assert result["tags_A"][1] == 0
        assert result["tags_B"][1] == 1
        assert result["tags_C"][1] == 1

        # Check third row: has A
        assert result["tags_A"][2] == 1
        assert result["tags_B"][2] == 0
        assert result["tags_C"][2] == 0

    def test_create_one_hot_preserves_original(self):
        """Test that original list column is preserved."""
        df = pl.DataFrame({"tags": [["A", "B"], ["B", "C"]]})
        transformation = Create_OneHotFromList(col="tags")

        result = transformation.transform(df)

        assert "tags" in result.columns


class TestCreate_BGGLink:
    """Test suite for Create_BGGLink transformation."""

    def test_create_bgg_link(self):
        """Test creating BGG link from ID."""
        df = pl.DataFrame({"id": ["123", "456", "789"]})
        transformation = Create_BGGLink(col="id")

        result = transformation.transform(df)

        assert "bgg_link" in result.columns
        assert result["bgg_link"][0] == "https://boardgamegeek.com/boardgame/123"
        assert result["bgg_link"][1] == "https://boardgamegeek.com/boardgame/456"
        assert result["bgg_link"][2] == "https://boardgamegeek.com/boardgame/789"


class TestTransform_ColumnTypes:
    """Test suite for Transform_ColumnTypes transformation."""

    def test_transform_column_types(self):
        """Test casting columns to specified types."""
        df = pl.DataFrame(
            {"int_col": ["1", "2", "3"], "float_col": ["1.5", "2.5", "3.5"], "str_col": [1, 2, 3]}
        )
        transformation = Transform_ColumnTypes(
            columns={"int_col": "int", "float_col": "float", "str_col": "str"}
        )

        result = transformation.transform(df)

        assert result.schema["int_col"] == pl.Int64
        assert result.schema["float_col"] == pl.Float64
        assert result.schema["str_col"] == pl.Utf8


class TestTransform_NormalizeColumn:
    """Test suite for Transform_NormalizeColumn transformation."""

    def test_normalize_column(self):
        """Test normalizing a column to [0, 1]."""
        df = pl.DataFrame({"values": [10, 20, 30], "filter": [1, 1, 1]})
        transformation = Transform_NormalizeColumn(col="values", filter_column="filter", value=1)

        result = transformation.transform(df)

        values = result["values"].to_list()
        assert values[0] == 0.0  # min
        assert values[1] == 0.5  # mid
        assert values[2] == 1.0  # max

    def test_normalize_with_filter(self):
        """Test normalization respects filter."""
        df = pl.DataFrame({"values": [10, 20, 30], "filter": [1, 0, 1]})
        transformation = Transform_NormalizeColumn(col="values", filter_column="filter", value=1)

        result = transformation.transform(df)

        # Middle row should keep original value
        assert result["values"][1] == 20


class TestTransform_ClipValues:
    """Test suite for Transform_ClipValues transformation."""

    def test_clip_values_max(self):
        """Test clipping values at maximum."""
        df = pl.DataFrame({"values": [1, 5, 10, 15]})
        transformation = Transform_ClipValues(col="values", max_val=10)

        result = transformation.transform(df)

        assert result["values"].to_list() == [1, 5, 10, 10]

    def test_clip_values_min(self):
        """Test clipping values at minimum."""
        df = pl.DataFrame({"values": [1, 5, 10, 15]})
        transformation = Transform_ClipValues(col="values", min_val=5)

        result = transformation.transform(df)

        assert result["values"].to_list() == [5, 5, 10, 15]

    def test_clip_values_both(self):
        """Test clipping values at both min and max."""
        df = pl.DataFrame({"values": [1, 5, 10, 15]})
        transformation = Transform_ClipValues(col="values", min_val=5, max_val=10)

        result = transformation.transform(df)

        assert result["values"].to_list() == [5, 5, 10, 10]


class TestUpdate_ColumnByThreshold:
    """Test suite for Update_ColumnByThreshold transformation."""

    def test_update_column_by_threshold(self):
        """Test updating filter column based on threshold."""
        df = pl.DataFrame({"ratings": [10, 50, 100], "filter": [1, 1, 1]})
        transformation = Update_ColumnByThreshold(
            col="ratings", threshold=30, filter_column="filter", value=0
        )

        result = transformation.transform(df)

        # Rows with ratings <= 30 should have filter = 0
        assert result["filter"].to_list() == [0, 1, 1]
