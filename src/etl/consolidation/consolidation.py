from src.etl.consolidation.abstract.transformations import (
    TransformationsManager,
    AddColumn,
    CastColumns,
    EmptyNullFilter,
    ThresholdColumnFilter,
    ThresholdListElementFilter,
    PlayerCountsEncode,
    NormalizeColumn,
    PopularityScoreCreate,
    OneHotEncodeListColumn
)
from src.other.abstract.write_reader_factory import WriterReaderFactory

from src.config import settings


def consolidation():
    raw_dir = settings.PATHS["raw_data"]
    writer_reader = WriterReaderFactory.create_from_directory(raw_dir)
    raw_data = writer_reader.read()
    
    print(f"Reading from: {writer_reader.file_path}")

    transformations = [
        CastColumns(columns={
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
        }),
        
        # Add recommendation flag column (1 = recommend, 0 = don't recommend)
        AddColumn(col="to_recommend", value=1),
        
        # Filter and flag based on ratings
        EmptyNullFilter(col="num_ratings"),
        ThresholdColumnFilter(col="num_ratings", threshold=30, filter_column="to_recommend", value=0),
        
        # Filter categories (list column)
        EmptyNullFilter(col="categories", is_list=True),
        ThresholdListElementFilter(col="categories", threshold=1000),
        
        # Filter mechanics (list column)
        EmptyNullFilter(col="mechanics", is_list=True),
        ThresholdListElementFilter(col="mechanics", threshold=81),
        
        # Process player counts
        EmptyNullFilter(col="min_players"),
        EmptyNullFilter(col="max_players"),
        PlayerCountsEncode(min_col="min_players", max_col="max_players", max_cap=20),
        
        # Normalize playing time
        EmptyNullFilter(col="playing_time"),
        NormalizeColumn(col="playing_time", filter_column="to_recommend", value=1),
        
        # Normalize min age
        EmptyNullFilter(col="min_age"),
        NormalizeColumn(col="min_age", filter_column="to_recommend", value=1),
        
        # Filter ratings and create popularity score
        EmptyNullFilter(col="bayesian_avg_rating"),
        PopularityScoreCreate(
            owned_col="owned_by", 
            wanted_col="wanted_by", 
            wished_col="wished_by",
            weights=(2.0, 1.0, 0.5), 
            filter_column="to_recommend", 
            value=1
        ),

        OneHotEncodeListColumn(col="categories"),

        OneHotEncodeListColumn(col="mechanics")
    ]

    pipeline = TransformationsManager(transformations=transformations)
    processed_data = pipeline.run_transformations(raw_data)

    output_dir = settings.PATHS["processed_data"]
    output_path = WriterReaderFactory.generate_output_path(output_dir, base_name="bgg")
    writer_reader.write(processed_data, output_path)
    
    print(f"Consolidation complete. Output written to: {output_path}")
    return processed_data


if __name__ == "__main__":
    consolidation()