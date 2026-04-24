import pytest
from cdisc_rules_engine.check_operators.dataframe_operators import DataframeType
from cdisc_rules_engine.models.dataset.dask_dataset import DaskDataset
from cdisc_rules_engine.models.dataset.pandas_dataset import PandasDataset


@pytest.mark.parametrize(
    "data,dataset_type,operator,expected_result",
    [
        (
            {
                "target": [["A", "B"], ["C", "D"], ["E", "F"]],
                "comparator": [["B", "C"], ["D", "E"], ["F", "G"]],
            },
            PandasDataset,
            "shares_at_least_one_element_with",
            [True, True, True],
        ),
        (
            {
                "target": [["A", "B"], ["C", "D"], ["E", "F"]],
                "comparator": [["X", "Y"], ["Y", "Z"], ["Z", "W"]],
            },
            DaskDataset,
            "shares_at_least_one_element_with",
            [False, False, False],
        ),
        (
            {
                "target": [["A"], ["B"], ["C"]],
                "comparator": [["A", "X"], ["B", "Y"], ["C", "Z"]],
            },
            PandasDataset,
            "shares_exactly_one_element_with",
            [True, True, True],
        ),
        (
            {
                "target": [["A", "B"], ["C", "D"], ["E", "F"]],
                "comparator": [["A", "X"], ["C", "Y"], ["E", "Z"]],
            },
            DaskDataset,
            "shares_exactly_one_element_with",
            [True, True, True],
        ),
        (
            {
                "target": [["A", "B"], ["C", "D"], ["E", "F"]],
                "comparator": [["X", "Y"], ["Y", "Z"], ["Z", "W"]],
            },
            PandasDataset,
            "shares_no_elements_with",
            [True, True, True],
        ),
        (
            {
                "target": [["A", "B"], ["C", "D"], ["E", "F"]],
                "comparator": [["B", "C"], ["D", "E"], ["F", "G"]],
            },
            DaskDataset,
            "shares_no_elements_with",
            [False, False, False],
        ),
    ],
)
def test_element_sharing_operators(data, dataset_type, operator, expected_result):
    df = dataset_type.from_dict(data)
    dataframe_type = DataframeType({"value": df})
    result = getattr(dataframe_type, operator)(
        {"target": "target", "comparator": "comparator"}
    )
    assert result.tolist() == expected_result


def test_element_sharing_operators_cases():
    data = {
        "target": [["A"], ["B", "C"], ["D", "E", "F"], []],
        "comparator": [["A", "B"], ["C", "D"], ["E", "F", "G"], ["X"]],
    }
    df = PandasDataset.from_dict(data)
    dataframe_type = DataframeType({"value": df})

    assert dataframe_type.shares_at_least_one_element_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [True, True, True, False]

    assert dataframe_type.shares_exactly_one_element_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [True, True, False, False]

    assert dataframe_type.shares_no_elements_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [False, False, False, True]


def test_element_sharing_operators_with_single_elements():
    data = {"target": ["A", "B", "C", "D"], "comparator": ["X", "B", "Y", "Z"]}
    df = DaskDataset.from_dict(data)
    dataframe_type = DataframeType({"value": df})

    assert dataframe_type.shares_at_least_one_element_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [False, True, False, False]

    assert dataframe_type.shares_exactly_one_element_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [False, True, False, False]

    assert dataframe_type.shares_no_elements_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [True, False, True, True]


def test_element_sharing_operators_with_mixed_types():
    data = {
        "target": [["A", "B"], "C", ["D", "E"], "F"],
        "comparator": ["B", ["C", "D"], "E", ["F", "G"]],
    }
    df = PandasDataset.from_dict(data)
    dataframe_type = DataframeType({"value": df})

    assert dataframe_type.shares_at_least_one_element_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [True, True, True, True]

    assert dataframe_type.shares_exactly_one_element_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [True, True, True, True]

    assert dataframe_type.shares_no_elements_with(
        {"target": "target", "comparator": "comparator"}
    ).tolist() == [False, False, False, False]


def test_shares_no_elements_with_literal_comparator_list():
    # Regression for ADAMCR-0053 / Finding #30. rules-adamig-1-3.json's
    # ADAMCR-0053 Check passes a literal list as the comparator
    # (e.g. shares_no_elements_with($datasets, ["DM"])). The operator used
    # to do row[comparator] unconditionally, which crashes when comparator
    # is a literal list rather than a column name:
    #   "None of [Index(['DM'], dtype='object')] are in the [index]".
    data = {
        "target": [["ADLB", "DM"], ["ADLB", "AE"]],
    }
    df = PandasDataset.from_dict(data)
    dataframe_type = DataframeType({"value": df})
    # Literal list comparator — operator must treat it as the literal value,
    # not a column to look up per row.
    assert dataframe_type.shares_no_elements_with(
        {"target": "target", "comparator": ["DM"]}
    ).tolist() == [False, True]


def test_shares_no_elements_with_post_merge_suffix_target():
    # After Match_Datasets merges, a column may be renamed (USUBJID → USUBJID.DM)
    # or suffixed (USUBJID_DM). The operator must fall back to those forms
    # rather than KeyError on the bare name.
    data = {
        "USUBJID.DM": [["001"], ["002"]],
    }
    df = PandasDataset.from_dict(data)
    dataframe_type = DataframeType({"value": df})
    # Target "USUBJID" does not exist as a column, but "USUBJID.DM" does.
    # Operator must resolve the dotted form and not crash.
    result = dataframe_type.shares_no_elements_with(
        {"target": "USUBJID", "comparator": ["001"]}
    ).tolist()
    assert result == [False, True]
