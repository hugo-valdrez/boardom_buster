from pathlib import Path

from src.other.abstract.base_write_reader import BaseWriteReader
from src.other.abstract.parquet_write_reader import ParqueWriteReader

class WriteReaderFactory:
    _readers = {
        ".parquet": ParqueWriteReader
    }

    @classmethod
    def create_write_reader(cls, file_path: Path) -> BaseWriteReader:
        extension = file_path.suffix.lower()

        if extension not in cls._readers:
            raise ValueError(
                f"Unsupported file extension: {extension}.\n"
                f"Supported file extensions: {', '.join(cls._readers.keys())}"
            )
        
        reader_class = cls._readers[extension]
        return reader_class(file_path)