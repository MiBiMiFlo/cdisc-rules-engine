from cdisc_rules_engine.check_operators.dataframe_operators import DataframeType
import pytest
from cdisc_rules_engine.models.dataset.dask_dataset import DaskDataset
from cdisc_rules_engine.models.dataset.pandas_dataset import PandasDataset


@pytest.mark.parametrize(
    "target, dataset_type, expected_result",
    [
        ("var1", PandasDataset, [True, True, True]),
        ("var1", DaskDataset, [True, True, True]),
        ("--r1", PandasDataset, [True, True, True]),
        ("--r1", DaskDataset, [True, True, True]),
        ("nested_var", PandasDataset, [True, True, True]),
        ("nested_var", DaskDataset, [True, True, True]),
        ("invalid", PandasDataset, [False, False, False]),
        ("invalid", DaskDataset, [False, False, False]),
        ("a", PandasDataset, [True, True, True]),
        ("a", DaskDataset, [True, True, True]),
        ("f", PandasDataset, [True, True, True]),
        ("f", DaskDataset, [True, True, True]),
        ("x", PandasDataset, [False, False, False]),
        ("x", DaskDataset, [False, False, False]),
        ("non_nested_value", PandasDataset, [True, True, True]),
        ("non_nested_value", DaskDataset, [True, True, True]),
    ],
)
def test_exists(target, dataset_type, expected_result):
    data = {
        "var1": [1, 2, 4],
        "var2": [3, 5, 6],
        "nested_var": [["a", "b", "c"], ["d", "e"], ["f", "nested_var", "g"]],
        "non_nested_value": ["h", "i", "j"],
    }
    df = dataset_type.from_dict(data)
    result = DataframeType({"value": df, "column_prefix_map": {"--": "va"}}).exists(
        {"target": target}
    )
    assert result.equals(df.convert_to_series(expected_result))


@pytest.mark.parametrize(
    "target, dataset_type, expected_result",
    [
        ("var1", PandasDataset, [False, False, False]),
        ("var1", DaskDataset, [False, False, False]),
        ("--r1", PandasDataset, [False, False, False]),
        ("--r1", DaskDataset, [False, False, False]),
        ("nested_var", PandasDataset, [False, False, False]),
        ("nested_var", DaskDataset, [False, False, False]),
        ("invalid", PandasDataset, [True, True, True]),
        ("invalid", DaskDataset, [True, True, True]),
        ("a", PandasDataset, [False, False, False]),
        ("a", DaskDataset, [False, False, False]),
        ("f", PandasDataset, [False, False, False]),
        ("f", DaskDataset, [False, False, False]),
        ("x", PandasDataset, [True, True, True]),
        ("x", DaskDataset, [True, True, True]),
        ("non_nested_value", PandasDataset, [False, False, False]),
        ("non_nested_value", DaskDataset, [False, False, False]),
    ],
)
def test_not_exists(target, dataset_type, expected_result):
    data = {
        "var1": [1, 2, 4],
        "var2": [3, 5, 6],
        "nested_var": [["a", "b", "c"], ["d", "e"], ["f", "nested_var", "g"]],
        "non_nested_value": ["h", "i", "j"],
    }
    df = dataset_type.from_dict(data)
    result = DataframeType({"value": df, "column_prefix_map": {"--": "va"}}).not_exists(
        {"target": target}
    )
    assert result.equals(df.convert_to_series(expected_result))


def test_exists_consults_original_columns_attr():
    # Regression for CORE-000012 / Finding #9. Variable Metadata Check rules
    # evaluate the Check against a metadata projection whose columns are
    # variable_name/variable_label/... - not the source dataset's columns.
    # The builder attaches _original_columns (frozenset of the source
    # dataset's column names) via df.attrs; exists must consult it so
    # `exists AEOCCUR` on an AE projection returns True when AEOCCUR was
    # a real column on the AE dataset.
    import pandas as pd
    meta_df = pd.DataFrame(
        {
            "variable_name": ["STUDYID", "USUBJID", "AETERM", "AEOCCUR"],
            "variable_label": ["", "", "", ""],
        }
    )
    meta_df.attrs["_original_columns"] = frozenset(
        ["STUDYID", "USUBJID", "AETERM", "AEOCCUR"]
    )
    df = PandasDataset(meta_df)

    # AEOCCUR is not a column on the projection but is in _original_columns.
    result = DataframeType({"value": df}).exists({"target": "AEOCCUR"})
    assert result.tolist() == [True, True, True, True]

    # A name that is neither a projection column nor in _original_columns
    # must return False.
    result = DataframeType({"value": df}).exists({"target": "NOTACOLUMN"})
    assert result.tolist() == [False, False, False, False]

    # The projection's meta columns themselves must still be findable via
    # the normal path, independent of _original_columns.
    result = DataframeType({"value": df}).exists({"target": "variable_name"})
    assert result.tolist() == [True, True, True, True]
