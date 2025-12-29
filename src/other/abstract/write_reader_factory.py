import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Type

from src.other.abstract.base_write_reader import BaseWriterReader
from src.other.abstract.parquet_write_reader import ParquetWriterReader


class WriterReaderFactory:
    """Factory for creating file reader/writer instances based on file extension."""

    _registry: Dict[str, Type[BaseWriterReader]] = {
        ".parquet": ParquetWriterReader,
    }

    DATE_PATTERN = re.compile(r"^(\d{8})_(.+)$")

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Return list of supported file extensions."""
        return list(cls._registry.keys())

    @classmethod
    def find_latest_file(cls, directory: Path, extension: str = ".parquet") -> Optional[Path]:
        """Find the most recent dated file in a directory.

        Expects files named like: YYYYMMDD_name.extension (e.g., 20250811_bgg.parquet)
        """
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        dated_files = []
        for file in directory.glob(f"*{extension}"):
            match = cls.DATE_PATTERN.match(file.stem)
            if match:
                date_str = match.group(1)
                try:
                    date = datetime.strptime(date_str, "%Y%m%d")
                    dated_files.append((date, file))
                except ValueError:
                    continue

        if not dated_files:
            return None

        dated_files.sort(key=lambda x: x[0], reverse=True)
        return dated_files[0][1]

    @classmethod
    def generate_output_path(
        cls, directory: Path, base_name: str = "bgg", extension: str = ".parquet"
    ) -> Path:
        """Generate an output file path with current date.

        Returns path like: directory/YYYYMMDD_name.extension
        """
        date = datetime.now()
        date_str = date.strftime("%Y%m%d")
        filename = f"{date_str}_{base_name}{extension}"
        return directory / filename

    @classmethod
    def create(cls, file_path: Path) -> BaseWriterReader:
        """Create a writer/reader instance for the given file path."""
        extension = file_path.suffix.lower()

        if extension not in cls._registry:
            raise ValueError(
                f"Unsupported file extension: '{extension}'. "
                f"Supported extensions: {cls.get_supported_extensions()}"
            )

        return cls._registry[extension](file_path)

    @classmethod
    def create_from_directory(
        cls, directory: Path, extension: str = ".parquet"
    ) -> BaseWriterReader:
        """Create a writer/reader for the latest dated file in a directory."""
        latest_file = cls.find_latest_file(directory, extension)

        if latest_file is None:
            raise FileNotFoundError(
                f"No dated {extension} files found in {directory}. "
                f"Expected format: YYYYMMDD_name{extension}"
            )

        return cls.create(latest_file)
