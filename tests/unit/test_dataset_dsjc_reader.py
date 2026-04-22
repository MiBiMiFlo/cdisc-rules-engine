import os

from cdisc_rules_engine.services.data_readers.dataset_dsjc_reader import (
    DatasetDSJCReader,
)


def test_from_file():
    test_dataset_path: str = (
        f"{os.path.dirname(__file__)}/../resources/test_dataset.dsjc"
    )

    reader = DatasetDSJCReader()
    dataframe = reader.from_file(test_dataset_path)
    for value in dataframe["EXDOSE"]:
        """
        Verify that the rounding of incredibly small values to 0 is applied.
        """
        assert value == 0 or abs(value) > 10**-16
