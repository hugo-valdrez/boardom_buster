from src.etl.consolidation.abstract.transformations import (
    TransformationsManager,
    Create_ConstantColumn,
    Transform_ColumnTypes,
    Filter_RowsNullEmpty,
    Update_ColumnByThreshold,
    Filter_ListElements,
    Create_PlayerCountColumns,
    Transform_NormalizeColumn,
    Create_PopularityScore,
    Create_OneHotFromList
)
from src.other.abstract.write_reader_factory import WriterReaderFactory

from src.config import settings


def consolidation():
    raw_dir = settings.PATHS["raw_data"]
    writer_reader = WriterReaderFactory.create_from_directory(raw_dir)
    raw_data = writer_reader.read()

    transformations = [
        Transform_ColumnTypes(columns={
            "num_ratings": "int",
            "min_players": "int",
            "max_players": "int",
            "playing_time": "float",
            "min_age": "int",
            "avg_rating": "float",
            "bayesian_avg_rating": "float",
            "owned_by": "int",
            "wanted_by": "int",
            "wished_by": "int",
            "publication_year": "int"
        }),
        
        # Add recommendation flag column (1 = recommend, 0 = don't recommend)
        Create_ConstantColumn(col="to_recommend", value=1),
        
        # Filter and flag based on ratings
        Filter_RowsNullEmpty(col="num_ratings"),
        Update_ColumnByThreshold(col="num_ratings", threshold=30, filter_column="to_recommend", value=0),
        
        # Filter categories (list column)
        Filter_RowsNullEmpty(col="categories", is_list=True),
        Filter_ListElements(col="categories", threshold=1000),
        
        # Filter mechanics (list column)
        Filter_RowsNullEmpty(col="mechanics", is_list=True),
        Filter_ListElements(col="mechanics", threshold=81),
        
        # Process player counts
        Filter_RowsNullEmpty(col="min_players"),
        Filter_RowsNullEmpty(col="max_players"),
        Create_PlayerCountColumns(min_col="min_players", max_col="max_players", max_cap=20),
        
        # Normalize playing time
        Filter_RowsNullEmpty(col="playing_time"),
        Transform_NormalizeColumn(col="playing_time", filter_column="to_recommend", value=1),
        
        # Normalize min age
        Filter_RowsNullEmpty(col="min_age"),
        Transform_NormalizeColumn(col="min_age", filter_column="to_recommend", value=1),
        
        # Filter ratings and create popularity score
        Filter_RowsNullEmpty(col="bayesian_avg_rating"),
        Create_PopularityScore(
            owned_col="owned_by", 
            wanted_col="wanted_by", 
            wished_col="wished_by",
            weights=(2.0, 1.0, 0.5), 
            filter_column="to_recommend", 
            value=1
        ),

        Create_OneHotFromList(col="categories"),

        Create_OneHotFromList(col="mechanics")
    ]

    pipeline = TransformationsManager(transformations=transformations)
    processed_data = pipeline.run_transformations(raw_data)

    output_dir = settings.PATHS["processed_data"]
    output_path = WriterReaderFactory.generate_output_path(output_dir, base_name="bgg", extension="parquet")
    writer_reader.write(processed_data, output_path)
    
    return processed_data


if __name__ == "__main__":
    consolidation()