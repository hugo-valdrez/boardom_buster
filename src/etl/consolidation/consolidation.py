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
    Create_OneHotFromList,
    Transform_ClipValues
)
from src.other.abstract.write_reader_factory import WriterReaderFactory

from src.config import settings


def consolidation():
    raw_dir = settings.PATHS["raw_data"]
    writer_reader = WriterReaderFactory.create_from_directory(raw_dir)
    raw_data = writer_reader.read()

    # Load configuration
    etl_config = settings.ETL["consolidation"]
    filters = etl_config["filters"]
    player_counts_config = etl_config["player_counts"]
    popularity_config = etl_config["popularity_score"]
    column_types = etl_config["column_types"]
    output_config = settings.ETL["output"]

    transformations = [
        Transform_ColumnTypes(columns=column_types),
        
        # Add recommendation flag column (1 = recommend, 0 = don't recommend)
        Create_ConstantColumn(col="to_recommend", value=1),
        
        # Filter and flag based on ratings
        Filter_RowsNullEmpty(col="num_ratings"),
        Update_ColumnByThreshold(
            col="num_ratings", 
            threshold=filters["num_ratings_threshold"], 
            filter_column="to_recommend", 
            value=0
        ),
        
        # Filter categories (list column)
        Filter_RowsNullEmpty(col="categories", is_list=True),
        Filter_ListElements(col="categories", threshold=filters["categories_threshold"]),
        
        # Filter mechanics (list column)
        Filter_RowsNullEmpty(col="mechanics", is_list=True),
        Filter_ListElements(col="mechanics", threshold=filters["mechanics_threshold"]),
        
        # Process player counts
        Filter_RowsNullEmpty(col="min_players"),
        Filter_RowsNullEmpty(col="max_players"),
        Create_PlayerCountColumns(
            min_col="min_players", 
            max_col="max_players", 
            max_cap=player_counts_config["max_cap"]
        ),
        
        # Normalize playing time
        Filter_RowsNullEmpty(col="playing_time"),
        Transform_ClipValues(col="playing_time", min_val=filters["min_playing_time"], max_val=filters["max_playing_time"]),
        # Transform_NormalizeColumn(col="playing_time", filter_column="to_recommend", value=1),
        
        # Filter publication year
        Filter_RowsNullEmpty(col="publication_year"),

        # Normalize min age
        Filter_RowsNullEmpty(col="min_age"),
        Transform_NormalizeColumn(col="min_age", filter_column="to_recommend", value=1),
        
        # Filter ratings and create popularity score
        Filter_RowsNullEmpty(col="bayesian_avg_rating"),
        Filter_RowsNullEmpty(col="avg_rating"),
        Create_PopularityScore(
            owned_col="owned_by", 
            wanted_col="wanted_by", 
            wished_col="wished_by",
            weights=(
                popularity_config["weight_owned"],
                popularity_config["weight_wanted"],
                popularity_config["weight_wished"]
            ), 
            filter_column="to_recommend", 
            value=1
        ),

        Create_OneHotFromList(col="categories"),

        Create_OneHotFromList(col="mechanics")
    ]

    pipeline = TransformationsManager(transformations=transformations)
    processed_data = pipeline.run_transformations(raw_data)

    output_dir = settings.PATHS["processed_data"]
    output_path = WriterReaderFactory.generate_output_path(
        output_dir, 
        base_name=output_config["base_filename"], 
        extension=output_config["file_extension"]
    )
    writer_reader.write(processed_data, output_path)
    
    return processed_data


if __name__ == "__main__":
    consolidation()