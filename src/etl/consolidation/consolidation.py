from src.etl.consolidation.abstract.transformations import *
from other.abstract.write_reader_factory import WriteReaderFactory
from pathlib import Path

from src.config import settings

def create_writer_reader(data_path: Path):
    return WriteReaderFactory.create_write_reader(data_path)


def consolidation():
    writer_reader = create_writer_reader(settings.PATHS["raw_data"])
    raw_data = writer_reader.read()

    transformations = [
        newcolumn_value_Add(col="to_recommend", value=1),
        emptynull_Filter(col="num_ratings"),
        threshold_column_Filter(col="num_ratings", threshold=30, filter_column="to_recommend", value=0),
        emptynull_Filter(col="categories"),
        threshold_listelement_Filter(col="categories", threshold=1000),
        emptynull_Filter(col="mechanics"),
        threshold_listelement_Filter(col="mechanics", threshold=81),
        emptynull_Filter(col="min_players"),
        emptynull_Filter(col="max_players"),
        playercounts_Encode(min_col="min_players", max_col="max_players", max_cap=20),
        emptynull_Filter(col="playing_time"),
        normalizecolumn_Transformation(col="playing_time", filter_column="to_recommend", value=1),
        emptynull_Filter(col="min_age"),
        normalizecolumn_Transformation(col="min_age", filter_column="to_recommend", value=1),
        emptynull_Filter(col="bayesian_avg_rating"),
        popularityscore_Create(owned_col="owned_by", wanted_col="wanted_by", wished_col="wished_by",
                               weights=(2.0, 1.0, 0.5), filter_column="to_recommend", value=1)
    ]

    pipeline = TransformationsManager(transformations=transformations)
    pipeline.run_transformations(raw_data)

    # read parquet file
    # run transformations
    # write new parquet
    
    pass

if __name__ == "__main__":
    consolidation()