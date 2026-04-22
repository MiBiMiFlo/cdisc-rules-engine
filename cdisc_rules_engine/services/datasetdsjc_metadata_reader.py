import zlib

from cdisc_rules_engine.services.datasetndjson_metadata_reader import (
    DatasetNDJSONMetadataReader,
)


class DatasetDSJCMetadataReader(DatasetNDJSONMetadataReader):
    """
    Responsibility of the class is to read metadata
    from a .dsjc file (zLib-compressed Dataset-NDJSON).
    """

    def _read_lines(self, encoding: str) -> list:
        try:
            with open(self._file_path, "rb") as file:
                raw = zlib.decompress(file.read())
        except zlib.error as e:
            raise ValueError(
                f"Could not decompress DSJC file {self._file_path}: {e}"
            )
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError) as e:
            raise ValueError(
                f"Could not decode DSJC file {self._file_path} with {encoding} encoding: {e}. "
                f"Please specify the correct encoding using the -e flag."
            )
        return text.splitlines(keepends=True)
