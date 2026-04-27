"""Regression tests for the ``has_multiple_values_for`` operator.

Fix #25 added a unidirectional functional-dependency operator to Python's
``DataframeType``, mirroring Java's ``OperatorRegistry.evalHasMultipleValuesFor``.
Affects ADAMCR-0693, 0694, 0732 (now also passing on Python after the
operator addition + multi-column ``within`` rule-body rewrite) and ADAMCR-0756,
0759 (Fix #23 spec ports — un-baselined here).

Operator semantics (mirrors Java):
- Flags rows whose ``comparator`` value maps to **more than one distinct**
  ``target`` value within the same partition.
- Unidirectional — does NOT also flag the inverse mapping (that's
  ``is_not_unique_relationship``'s bidirectional behaviour).
- ``within`` is polymorphic: ``"PARAMCD"`` for single-column, or
  ``["USUBJID", "PARAMCD"]`` for multi-column partitioning. Missing/empty
  ``within`` means ungrouped (whole frame).
- Rows with missing or empty target or comparator cells are excluded from
  grouping (do not contribute to the per-key value set).
"""
import pandas as pd
import pytest

from cdisc_rules_engine.check_operators.dataframe_operators import DataframeType
from cdisc_rules_engine.models.dataset.pandas_dataset import PandasDataset


def _df(rows):
    """Build a DataframeType wrapping a PandasDataset with the given rows."""
    df = pd.DataFrame(rows)
    return DataframeType({"value": PandasDataset(df), "column_prefix_map": {}})


# ---------------------------------------------------------------------------
# Core semantic — unidirectional, single-column within
# ---------------------------------------------------------------------------


def test_unidirectional_flags_only_offending_comparator_keys():
    """Flag rows whose comparator value maps to multiple distinct targets;
    do NOT flag the inverse direction (that's is_not_unique_relationship).
    """
    op = _df([
        {"BASEC": "Sitting",  "BASE": "60"},
        {"BASEC": "Sitting",  "BASE": "65"},   # bad: Sitting → 60, 65
        {"BASEC": "Standing", "BASE": "75"},
        {"BASEC": "Standing", "BASE": "75"},
    ])
    result = op.has_multiple_values_for({"target": "BASE", "comparator": "BASEC"}).tolist()
    assert result == [True, True, False, False], (
        "BASEC=Sitting maps to two BASE values; both Sitting rows fire. "
        "Standing rows are consistent — no flag."
    )


def test_unidirectional_inverse_is_not_flagged():
    """If BASE=60 maps to two BASEC values, the inverse direction should NOT
    fire (the operator is unidirectional)."""
    op = _df([
        {"BASEC": "A", "BASE": "60"},
        {"BASEC": "B", "BASE": "60"},   # BASE=60 maps to two BASEC, but
                                         # we only check BASEC → BASE.
        {"BASEC": "A", "BASE": "60"},
    ])
    result = op.has_multiple_values_for({"target": "BASE", "comparator": "BASEC"}).tolist()
    # BASEC=A maps to BASE={60} — single, no fire.
    # BASEC=B maps to BASE={60} — single, no fire.
    assert result == [False, False, False]


def test_single_column_within_partitions_per_group():
    """Each PARAMCD group is evaluated independently; clean groups don't
    propagate into the violation set of a dirty one."""
    op = _df([
        {"PARAMCD": "HR",  "BASEC": "Sitting", "BASE": "60"},
        {"PARAMCD": "HR",  "BASEC": "Sitting", "BASE": "65"},   # dirty
        {"PARAMCD": "BMI", "BASEC": "Normal",  "BASE": "22"},
        {"PARAMCD": "BMI", "BASEC": "Normal",  "BASE": "22"},   # clean
    ])
    result = op.has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC", "within": "PARAMCD"}
    ).tolist()
    assert result == [True, True, False, False], (
        "HR group has Sitting→{60,65} (dirty); BMI group is clean"
    )


# ---------------------------------------------------------------------------
# Multi-column within (Fix #25)
# ---------------------------------------------------------------------------


def test_multi_column_within_disjoint_subjects_consistent():
    """ADAMCR-0693 / 0694 shape: ``within: [USUBJID, SPDEVID, PARAMCD]``.
    Two subjects with internally-consistent but mutually-different mappings
    must NOT cross-flag each other — the multi-column partition keeps
    subjects independent.
    """
    op = _df([
        {"USUBJID": "S1", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Sitting",  "BASE": "60"},
        {"USUBJID": "S1", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Sitting",  "BASE": "60"},
        {"USUBJID": "S2", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Standing", "BASE": "75"},
        {"USUBJID": "S2", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Standing", "BASE": "75"},
    ])
    result = op.has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC",
         "within": ["USUBJID", "SPDEVID", "PARAMCD"]}
    ).tolist()
    assert result == [False, False, False, False], (
        "Each (USUBJID, SPDEVID, PARAMCD) group is internally consistent; "
        "different subjects can legitimately have different mappings."
    )


def test_multi_column_within_flags_only_offending_group():
    """Only the group with the inconsistent mapping fires."""
    op = _df([
        {"USUBJID": "S1", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Sitting", "BASE": "60"},
        {"USUBJID": "S1", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Sitting", "BASE": "65"},   # offending
        {"USUBJID": "S2", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Sitting", "BASE": "75"},
        {"USUBJID": "S2", "SPDEVID": "D1", "PARAMCD": "HR",
         "BASEC": "Sitting", "BASE": "75"},
    ])
    result = op.has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC",
         "within": ["USUBJID", "SPDEVID", "PARAMCD"]}
    ).tolist()
    assert result == [True, True, False, False], (
        "S1 group has Sitting→{60,65}; S2 group is clean. The offending "
        "group's BASEC=Sitting must NOT propagate into S2's clean rows."
    )


def test_single_element_list_equivalent_to_string():
    """``within: ["PARAMCD"]`` (list of one) must behave identically to
    ``within: "PARAMCD"`` — backward-compat with the polymorphic field.
    """
    rows = [
        {"PARAMCD": "HR", "BASEC": "Sitting", "BASE": "60"},
        {"PARAMCD": "HR", "BASEC": "Sitting", "BASE": "65"},
    ]
    string_form = _df(rows).has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC", "within": "PARAMCD"}
    ).tolist()
    list_form = _df(rows).has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC", "within": ["PARAMCD"]}
    ).tolist()
    assert string_form == list_form == [True, True]


# ---------------------------------------------------------------------------
# Missing / empty cell handling — mirrors Java's isMissing() skip
# ---------------------------------------------------------------------------


def test_missing_target_or_comparator_excluded():
    """Rows with missing target OR comparator are excluded from grouping
    entirely — they can't be flagged because they don't contribute to
    the per-key value set."""
    op = _df([
        {"BASEC": "Sitting", "BASE": "60"},
        {"BASEC": "Sitting", "BASE": None},     # missing target — excluded
        {"BASEC": None,      "BASE": "70"},     # missing comparator — excluded
        {"BASEC": "Sitting", "BASE": "60"},
    ])
    result = op.has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC"}
    ).tolist()
    # BASEC=Sitting → BASE values from rows 0, 3 = {60} — single, no fire.
    assert result == [False, False, False, False]


def test_complement_does_not_have_multiple_values_for():
    """The inverse operator returns the bitwise complement."""
    op = _df([
        {"BASEC": "Sitting", "BASE": "60"},
        {"BASEC": "Sitting", "BASE": "65"},
        {"BASEC": "Standing", "BASE": "75"},
    ])
    affirmative = op.has_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC"}
    ).tolist()
    negative = op.does_not_have_multiple_values_for(
        {"target": "BASE", "comparator": "BASEC"}
    ).tolist()
    # Both Sitting rows fire affirmative; Standing row doesn't.
    assert affirmative == [True, True, False]
    assert negative == [False, False, True]


# ---------------------------------------------------------------------------
# Edge — empty/no-target safe defaults
# ---------------------------------------------------------------------------


def test_no_target_or_comparator_returns_all_false():
    """Defensive: if the leaf is malformed (missing target/comparator),
    return all-False rather than crashing."""
    op = _df([{"BASEC": "X", "BASE": "1"}])
    assert op.has_multiple_values_for({"comparator": "BASEC"}).tolist() == [False]
    assert op.has_multiple_values_for({"target": "BASE"}).tolist() == [False]
