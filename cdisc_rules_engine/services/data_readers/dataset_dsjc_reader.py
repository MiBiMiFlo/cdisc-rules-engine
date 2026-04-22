import json
import zlib

from cdisc_rules_engine.services.data_readers.dataset_ndjson_reader import (
    DatasetNDJSONReader,
)


class DatasetDSJCReader(DatasetNDJSONReader):

    def read_json_file(self, file_path: str) -> dict:
        try:
            with open(file_path, "rb") as file:
                raw = zlib.decompress(file.read())
            text = raw.decode(self.encoding)
        except zlib.error as e:
            raise ValueError(f"Could not decompress DSJC file {file_path}: {e}")
        except (UnicodeDecodeError, UnicodeError) as e:
            raise ValueError(
                f"Could not decode DSJC file {file_path} with {self.encoding} encoding: {e}. "
                f"Please specify the correct encoding using the -e flag."
            )
        lines = text.splitlines()
        return json.loads(lines[0]), [json.loads(line) for line in lines[1:] if line]
