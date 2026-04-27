"""Regression tests for CORE-000597 — Symptom B / Finding #24:
``QNAM equal_to <X>`` Check leaves on rules that pivot-merge SUPPAE.

The rule fires when a SUPPAE child has ``QNAM == "AESOSP"`` AND the parent
AE row's ``AESMIE != "Y"``. ``DataProcessor.merge_pivot_supp_dataset`` ->
``process_supp`` pivots SUPPAE so each unique QNAM becomes its own column
on the merged AE frame, and the original ``QNAM``/``QVAL``/``QLABEL``
columns are dropped. Without the preprocessor's QNAM-leaf rewrite, the
``business_rules`` evaluator raises ``KeyError`` when looking up the
``QNAM`` column. The fix:

1. ``DatasetPreprocessor`` records the QNAMs about to be pivoted right
   before ``merge_pivot_supp_dataset`` is called.
2. After all merges complete, it walks the rule's conditions and rewrites
   ``{name: "get_dataset", operator: "equal_to",
   value: {target: "QNAM", comparator: "AESOSP", ...}}`` leaves to
   ``{name: "get_dataset", operator: "non_empty",
   value: {target: "AESOSP"}}`` — the post-pivot semantic equivalent.
3. ``QNAM`` and ``QVAL`` columns are synthesized on the merged frame so
   sibling QVAL leaves and ``Outcome.Output_Variables`` entries naming
   QNAM/QVAL still resolve.

Mirrors Java's leaf-evaluator fallback that resolves unqualified ``QNAM``
against the joined SUPPAE dataset; both engines now emit the same
canonical violation shape.
"""
from unittest.mock import MagicMock

import pandas as pd

from cdisc_rules_engine.dummy_models.dummy_dataset import DummyDataset
from cdisc_rules_engine.models.dataset.pandas_dataset import PandasDataset
from cdisc_rules_engine.models.rule_conditions.condition_composite_factory import (
    ConditionCompositeFactory,
)
from cdisc_rules_engine.models.sdtm_dataset_metadata import SDTMDatasetMetadata
from cdisc_rules_engine.services.cache.in_memory_cache_service import (
    InMemoryCacheService,
)
from cdisc_rules_engine.services.data_services import DummyDataService
from cdisc_rules_engine.utilities.dataset_preprocessor import DatasetPreprocessor


def _build_preprocessor(ae_rows: list, suppae_rows: list):
    """Build a DatasetPreprocessor over an AE primary + SUPPAE child
    fixture. The DummyDataService's ``get_dataset`` uses the dataset's
    filename to look up; we stage AE as the primary and SUPPAE as the
    child to be merged in."""
    ae = DummyDataset({
        "name": "AE",
        "filename": "ae.xpt",
        "label": "Adverse Events",
        "file_size": 1000,
        "records": {k: [r.get(k) for r in ae_rows]
                    for k in {k for r in ae_rows for k in r}},
    })
    suppae = DummyDataset({
        "name": "SUPPAE",
        "filename": "suppae.xpt",
        "label": "SUPP for AE",
        "file_size": 1000,
        "records": {k: [r.get(k) for r in suppae_rows]
                    for k in {k for r in suppae_rows for k in r}},
    })
    cache = InMemoryCacheService()
    svc = DummyDataService(cache, MagicMock(), MagicMock(), data=[ae, suppae])
    primary = PandasDataset(ae.data.copy())
    primary_meta = SDTMDatasetMetadata(
        name="AE",
        first_record={"DOMAIN": "AE"},
        filename="ae.xpt",
        full_path="ae.xpt",
    )
    return DatasetPreprocessor(primary, primary_meta, svc, cache), [
        primary_meta,
        SDTMDatasetMetadata(
            name="SUPPAE",
            first_record={"DOMAIN": "SUPPAE", "RDOMAIN": "AE"},
            filename="suppae.xpt",
            full_path="suppae.xpt",
        ),
    ]


def _make_rule_with_qnam_check():
    """Build CORE-000597-shaped rule dict (post-``Rule.from_cdisc_metadata``
    deserialisation): ``QNAM equal_to "AESOSP" AND AESMIE not_equal_to "Y"``,
    Match_Datasets: SUPPAE on USUBJID."""
    conditions_dict = {
        "all": [
            {
                "name": "get_dataset",
                "operator": "equal_to",
                "value": {
                    "target": "QNAM",
                    "comparator": "AESOSP",
                    "value_is_literal": True,
                },
            },
            {
                "name": "get_dataset",
                "operator": "not_equal_to",
                "value": {
                    "target": "AESMIE",
                    "comparator": "Y",
                    "value_is_literal": True,
                },
            },
        ]
    }
    return {
        "core_id": "CORE-000597",
        "rule_type": "Record Data",
        "sensitivity": "Record",
        "datasets": [
            {"domain_name": "SUPPAE", "match_key": ["USUBJID"], "wildcard": "**"}
        ],
        "conditions": ConditionCompositeFactory.get_condition_composite(
            conditions_dict
        ),
    }


# ---------------------------------------------------------------------------
# Fix 1 — preprocess captures pivoted QNAMs and rewrites equal_to leaves
# ---------------------------------------------------------------------------


def test_qnam_equal_to_leaf_rewritten_after_supp_pivot():
    """The Check leaf ``QNAM equal_to "AESOSP"`` should be rewritten to
    ``AESOSP non_empty`` after the SUPP pivot drops the QNAM column.
    Before the fix the leaf survived as-is and ``business_rules`` raised
    KeyError when looking up the (now-missing) QNAM column.
    """
    pp, metas = _build_preprocessor(
        ae_rows=[
            {"STUDYID": "S1", "DOMAIN": "AE", "USUBJID": "001",
             "AESEQ": 1, "AETERM": "X", "AESMIE": "N"},
        ],
        suppae_rows=[
            {"STUDYID": "S1", "RDOMAIN": "AE", "USUBJID": "001",
             "IDVAR": "", "IDVARVAL": "",
             "QNAM": "AESOSP", "QLABEL": "x", "QVAL": "detail-001"},
        ],
    )
    rule = _make_rule_with_qnam_check()
    pp.preprocess(rule, metas)

    leaves = list(rule["conditions"].values())
    qnam_leaves = [
        leaf for leaf in leaves
        if isinstance(leaf.get("value"), dict)
        and leaf["value"].get("target") in ("QNAM", "AESOSP")
    ]
    assert len(qnam_leaves) == 1, (
        f"expected exactly one rewritten leaf, got {qnam_leaves!r}"
    )
    leaf = qnam_leaves[0]
    assert leaf["operator"] == "non_empty", (
        f"leaf should be rewritten to non_empty, got operator={leaf.get('operator')!r}"
    )
    assert leaf["value"] == {"target": "AESOSP"}, (
        f"leaf value should target the pivoted column, got {leaf.get('value')!r}"
    )
    assert "comparator" not in leaf["value"], (
        "comparator should be dropped from rewritten leaf"
    )


# ---------------------------------------------------------------------------
# Fix 2 — synthesized QNAM/QVAL columns on the merged frame
# ---------------------------------------------------------------------------


def test_qnam_qval_columns_synthesized_after_pivot():
    """After the SUPP pivot, ``QNAM`` and ``QVAL`` columns are dropped by
    ``process_supp``. The preprocessor now adds them back, populating
    each row with the matched-QNAM literal and its pivoted value, so
    sibling QVAL Check leaves and ``Outcome.Output_Variables`` lists
    naming QNAM/QVAL resolve correctly.
    """
    pp, metas = _build_preprocessor(
        ae_rows=[
            {"STUDYID": "S1", "DOMAIN": "AE", "USUBJID": "001",
             "AESEQ": 1, "AETERM": "X", "AESMIE": "N"},
            {"STUDYID": "S1", "DOMAIN": "AE", "USUBJID": "002",
             "AESEQ": 1, "AETERM": "Y", "AESMIE": "Y"},
            {"STUDYID": "S1", "DOMAIN": "AE", "USUBJID": "003",
             "AESEQ": 1, "AETERM": "Z", "AESMIE": "N"},
        ],
        suppae_rows=[
            {"STUDYID": "S1", "RDOMAIN": "AE", "USUBJID": "001",
             "IDVAR": "", "IDVARVAL": "",
             "QNAM": "AESOSP", "QLABEL": "x", "QVAL": "detail-001"},
            {"STUDYID": "S1", "RDOMAIN": "AE", "USUBJID": "002",
             "IDVAR": "", "IDVARVAL": "",
             "QNAM": "AESOSP", "QLABEL": "x", "QVAL": "detail-002"},
            {"STUDYID": "S1", "RDOMAIN": "AE", "USUBJID": "003",
             "IDVAR": "", "IDVARVAL": "",
             "QNAM": "AESPID", "QLABEL": "x", "QVAL": "SP003"},
        ],
    )
    rule = _make_rule_with_qnam_check()
    result = pp.preprocess(rule, metas)

    cols = list(result.columns)
    assert "AESOSP" in cols, "pivoted QNAM column must remain after merge"
    assert "QNAM" in cols, "QNAM column must be synthesized post-merge"
    assert "QVAL" in cols, "QVAL column must be synthesized post-merge"

    # Row 0 (USUBJID=001) and row 1 (USUBJID=002) had QNAM=AESOSP in SUPPAE;
    # row 2 (USUBJID=003) had QNAM=AESPID, which is NOT one of the rewritten
    # QNAMs (the rule's Check only matched AESOSP), so QNAM/QVAL stay NA.
    qnam = result["QNAM"].tolist()
    qval = result["QVAL"].tolist()
    assert qnam[0] == "AESOSP" and qnam[1] == "AESOSP"
    assert pd.isna(qnam[2]), (
        "row with non-matching QNAM should get NA — only QNAMs matched by the "
        "Check's equal_to leaves seed the synthesized column"
    )
    assert qval[0] == "detail-001" and qval[1] == "detail-002"
    assert pd.isna(qval[2])


# ---------------------------------------------------------------------------
# Fix 3 — leaves not matching the QNAM-equal_to pattern are left alone
# ---------------------------------------------------------------------------


def test_non_qnam_leaves_untouched():
    """Only leaves matching the exact pre-pivot ``QNAM equal_to <X>`` shape
    where ``<X>`` is a pivoted QNAM should be rewritten. Sibling leaves on
    other columns (e.g. AESMIE) must survive unchanged so the violation's
    full shape is preserved.
    """
    pp, metas = _build_preprocessor(
        ae_rows=[
            {"STUDYID": "S1", "DOMAIN": "AE", "USUBJID": "001",
             "AESEQ": 1, "AETERM": "X", "AESMIE": "N"},
        ],
        suppae_rows=[
            {"STUDYID": "S1", "RDOMAIN": "AE", "USUBJID": "001",
             "IDVAR": "", "IDVARVAL": "",
             "QNAM": "AESOSP", "QLABEL": "x", "QVAL": "v"},
        ],
    )
    rule = _make_rule_with_qnam_check()
    pp.preprocess(rule, metas)

    leaves = list(rule["conditions"].values())
    aesmie_leaves = [
        leaf for leaf in leaves
        if isinstance(leaf.get("value"), dict)
        and leaf["value"].get("target") == "AESMIE"
    ]
    assert len(aesmie_leaves) == 1
    leaf = aesmie_leaves[0]
    assert leaf["operator"] == "not_equal_to", (
        f"AESMIE leaf must not be rewritten; got operator={leaf.get('operator')!r}"
    )
    assert leaf["value"]["comparator"] == "Y"


# ---------------------------------------------------------------------------
# Fix 4 — no-SUPP rule passes through unchanged
# ---------------------------------------------------------------------------


def test_no_supp_join_no_qnam_rewrite():
    """A rule with no SUPP/SQ Match_Datasets must leave the Check
    untouched. The QNAM rewrite is gated on ``self._pivoted_qnams`` which
    only accumulates inside the SUPP merge branch.
    """
    ae = DummyDataset({
        "name": "AE",
        "filename": "ae.xpt",
        "label": "Adverse Events",
        "file_size": 1000,
        "records": {
            "STUDYID": ["S1"], "DOMAIN": ["AE"], "USUBJID": ["001"],
            "QNAM": ["X"],  # an AE column literally named QNAM (unrealistic
                            # but stresses that the rewrite isn't triggered
                            # without a SUPP merge)
        },
    })
    cache = InMemoryCacheService()
    svc = DummyDataService(cache, MagicMock(), MagicMock(), data=[ae])
    primary = PandasDataset(ae.data.copy())
    primary_meta = SDTMDatasetMetadata(
        name="AE", first_record={"DOMAIN": "AE"},
        filename="ae.xpt", full_path="ae.xpt",
    )
    pp = DatasetPreprocessor(primary, primary_meta, svc, cache)
    # Rule with no datasets ⇒ preprocess returns immediately without ever
    # accumulating pivoted QNAMs.
    rule = {
        "core_id": "TEST",
        "datasets": None,
        "conditions": ConditionCompositeFactory.get_condition_composite({
            "all": [{
                "name": "get_dataset",
                "operator": "equal_to",
                "value": {"target": "QNAM", "comparator": "AESOSP",
                          "value_is_literal": True},
            }]
        }),
    }
    pp.preprocess(rule, [primary_meta])
    leaf = list(rule["conditions"].values())[0]
    assert leaf["operator"] == "equal_to", (
        "no SUPP merge happened, the leaf should be untouched"
    )
    assert leaf["value"].get("comparator") == "AESOSP"
