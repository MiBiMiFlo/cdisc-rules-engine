"""Regression tests for CORE-000598 — ``Dataset Metadata Check`` rule
``dataset_name prefix_not_equal_to DOMAIN`` (SDTMIG 3.4 / 3.3 / 3.2).

The rule fires when a dataset's name does not start with the value its first
``DOMAIN`` cell carries (it codifies the SDTM convention "the dataset name
must begin with the DOMAIN value"). Three engine-side problems used to keep
the rule from emitting a usable violation, all of which are exercised here:

1. ``BaseDataService.get_dataset_metadata`` returned a frame with only a
   lowercase ``domain`` column. The rule's ``Output_Variables`` list the
   SDTM ``DOMAIN`` (uppercase) variable; without an uppercase column the
   ``COREActions`` error generator wrote the literal sentinel
   ``"Not in dataset"`` instead of the actual DOMAIN value.

2. ``RulesEngine.validate_single_rule`` keyed its result dict by
   ``dataset_metadata.unsplit_name``. For non-split datasets the
   ``unsplit_name`` property reads ``first_record["DOMAIN"]`` — when the
   DOMAIN cell disagrees with the dataset name (the very thing CORE-000598
   detects), the result key was that disagreeing value, not the dataset
   name. External callers that look results up by dataset name (e.g. a
   parity harness, the spec runner) never saw the violation.

3. ``DummyDataService.get_raw_dataset_metadata`` rebuilt the
   ``SDTMDatasetMetadata`` from a path-based metadata dict and hardcoded
   ``first_record={"DOMAIN": dataset_name}``, ignoring the actual mock
   ``records`` the test author supplied. Under the dummy service the rule
   could never observe a DOMAIN ≠ dataset_name configuration, so the
   regression couldn't even be reproduced through the dummy harness.

Each test below pins one of the three behaviours.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from cdisc_rules_engine.dummy_models.dummy_dataset import DummyDataset
from cdisc_rules_engine.models.dataset.pandas_dataset import PandasDataset
from cdisc_rules_engine.models.sdtm_dataset_metadata import SDTMDatasetMetadata
from cdisc_rules_engine.services.data_services import (
    DummyDataService,
    LocalDataService,
)


# ---------------------------------------------------------------------------
# Fix 1 — BaseDataService exposes both ``domain`` and ``DOMAIN`` columns
# ---------------------------------------------------------------------------


@patch("cdisc_rules_engine.services.data_services.LocalDataService.read_metadata")
def test_get_dataset_metadata_exposes_uppercase_DOMAIN(mock_read_metadata):
    """The metadata frame must carry an SDTM-cased ``DOMAIN`` column.

    CORE-000598's ``Output_Variables: ["dataset_name", "DOMAIN"]`` resolves
    against this frame. Without a column literally named ``DOMAIN`` the
    error generator emits the sentinel ``"Not in dataset"`` for that key,
    which downstream parity tooling has to special-case (and which Java
    never produces, since Java reads DOMAIN from the actual dataset).
    """
    mock_read_metadata.return_value = {
        "file_metadata": {
            "path": "study/ae.xpt",
            "name": "ae.xpt",
            "file_size": 12345,
        },
        "contents_metadata": {
            "dataset_label": "Adverse Events",
            "dataset_name": "AE",
            # The DOMAIN cell deliberately disagrees with the dataset name —
            # this is the configuration CORE-000598 is meant to detect.
            "first_record": {"DOMAIN": "XY"},
            "dataset_modification_date": datetime.now().isoformat(),
            "dataset_length": 1,
        },
    }
    cache = MagicMock()
    cache.get_dataset = lambda key: None
    cache.get = lambda key: None

    ds = LocalDataService(cache, MagicMock(), MagicMock())
    md: PandasDataset = ds.get_dataset_metadata(dataset_name="study/ae.xpt")

    cols = list(md.data.columns)
    assert "domain" in cols, "lowercase alias must remain (no regression on existing rules)"
    assert "DOMAIN" in cols, "SDTM-cased DOMAIN column missing — Output_Variables can't resolve"

    row = md.data.iloc[0].to_dict()
    assert row["domain"] == "XY"
    assert row["DOMAIN"] == "XY", (
        "DOMAIN must reflect the first-record DOMAIN cell, not the dataset name "
        "(CORE-000598 needs to surface the disagreeing value)"
    )


# ---------------------------------------------------------------------------
# Fix 2 — DummyDataService preserves the actual ``first_record``
# ---------------------------------------------------------------------------


def test_dummy_data_service_preserves_first_record_DOMAIN():
    """``DummyDataService.get_raw_dataset_metadata`` must keep the original
    ``first_record`` from the mock dataset, not synthesise
    ``{"DOMAIN": dataset_name}``.

    Without this fix, every rule that consults ``dataset_metadata.domain``
    in the dummy harness sees the dataset name, so the disagree-with-DOMAIN
    configuration that CORE-000598 catches is unreachable through the dummy
    service — the rule's whole point can't be unit-tested.
    """
    dataset_data = [
        {
            "name": "AE",
            "filename": "ae.xpt",
            "label": "Adverse Events",
            "file_size": 2000,
            "records": {
                "STUDYID": ["S1"],
                "DOMAIN": ["XY"],  # disagreement with the dataset name
                "USUBJID": ["001"],
                "AESEQ": [1],
            },
        }
    ]
    datasets = [DummyDataset(d) for d in dataset_data]
    cache = MagicMock()
    cache.get_dataset = lambda key: None
    cache.get = lambda key: None
    svc = DummyDataService(cache, MagicMock(), MagicMock(), data=datasets)

    raw = svc.get_raw_dataset_metadata(dataset_name="ae.xpt")
    assert raw.name == "AE"
    assert raw.first_record.get("DOMAIN") == "XY", (
        "DummyDataService dropped the mock DOMAIN cell and synthesised "
        "DOMAIN=dataset_name; CORE-000598 disagreement scenario can't be "
        "tested through this service."
    )
    # The downstream metadata frame should now carry the same disagreeing value.
    md = svc.get_dataset_metadata(dataset_name="ae.xpt", datasets=datasets)
    assert md.data.iloc[0]["DOMAIN"] == "XY"


# ---------------------------------------------------------------------------
# Fix 3 — ``validate_single_rule`` keys results by dataset name
# ---------------------------------------------------------------------------


def test_validate_single_rule_keys_results_by_dataset_name():
    """Result-dict keys must be ``dataset_metadata.name``, not
    ``unsplit_name``.

    For a non-split dataset whose DOMAIN cell disagrees with its name (the
    very configuration CORE-000598 detects), ``unsplit_name`` returns the
    DOMAIN cell — losing the dataset identity. Callers that index the
    return dict by the dataset name (parity harnesses, the spec runner)
    would silently miss the violation. Split-dataset filtering still works
    because the engine tracks ``unsplit_name`` parents in a separate set.
    """
    from cdisc_rules_engine.rules_engine import RulesEngine
    from cdisc_rules_engine.models.rule import Rule
    from cdisc_rules_engine.services.cache.in_memory_cache_service import (
        InMemoryCacheService,
    )

    rule_dict = {
        "Core": {"Id": "CORE-000598", "Status": "Published", "Version": "1"},
        "Rule_Type": "Dataset Metadata Check",
        "Sensitivity": "Dataset",
        "Executability": "Fully Executable",
        "Authorities": [
            {"Organization": "CDISC", "Standards": [
                {"Name": "SDTMIG", "References": [{
                    "Citations": [{"Cited_Guidance": "test", "Document": "x", "Section": "x"}],
                    "Origin": "test", "Rule_Identifier": {"Id": "x", "Version": "1"},
                    "Version": "2.0"}], "Version": "3.4"}]}
        ],
        "Scope": {"Classes": {"Include": ["ALL"]}, "Domains": {}},
        "Check": {"all": [
            {"name": "dataset_name", "operator": "prefix_not_equal_to",
             "prefix": 2, "value": "DOMAIN"}
        ]},
        "Outcome": {
            "Message": "Dataset name does not begin with DOMAIN value",
            "Output_Variables": ["dataset_name", "DOMAIN"],
        },
    }
    rule = Rule.from_cdisc_metadata(rule_dict)

    dataset = [
        {
            "name": "AE",
            "filename": "ae.xpt",
            "label": "Adverse Events",
            "file_size": 1000,
            "records": {
                "STUDYID": ["S1"],
                "DOMAIN": ["XY"],
                "USUBJID": ["001"],
                "AESEQ": [1],
            },
        }
    ]
    datasets = [DummyDataset(d) for d in dataset]
    cache = InMemoryCacheService()
    svc = DummyDataService(cache, MagicMock(), MagicMock(), data=datasets)
    metas = [svc.get_raw_dataset_metadata(dataset_name="ae.xpt")]

    engine = RulesEngine(
        data_service=svc, cache=cache, config_obj=MagicMock(),
        standard="sdtmig", standard_version="3.4",
    )
    results = engine.validate_single_rule(rule, metas)

    assert "AE" in results, (
        "result key should be dataset_metadata.name (\"AE\"); got "
        f"{list(results.keys())} — CORE-000598 disagreement scenario "
        "regressed back to keying by unsplit_name (the DOMAIN cell value)"
    )
    assert "XY" not in results, (
        "DOMAIN cell value leaked into the result-dict key — split-dataset "
        "tracking should be decoupled from result keys"
    )
    # The rule actually fires (sanity).
    entries = results["AE"]
    assert any(
        e.get("executionStatus") == "issue reported" for e in entries
    ), "CORE-000598 should fire at least one violation for the AE/XY mismatch"


# ---------------------------------------------------------------------------
# Fix 1 cross-check — Output_Variables resolves DOMAIN to the cell value,
# not the "Not in dataset" sentinel.
# ---------------------------------------------------------------------------


def test_dataset_metadata_check_emits_actual_DOMAIN_value():
    """End-to-end: a rule with ``Output_Variables: ["DOMAIN"]`` running over
    an AE dataset whose DOMAIN cell is ``XY`` must surface ``XY`` (not the
    ``"Not in dataset"`` sentinel) in the violation's ``value`` map.

    This is the user-visible payoff of fixes 1+2+3 together; if any one of
    them regresses, the assertion below fails.
    """
    from cdisc_rules_engine.rules_engine import RulesEngine
    from cdisc_rules_engine.models.rule import Rule
    from cdisc_rules_engine.services.cache.in_memory_cache_service import (
        InMemoryCacheService,
    )

    rule_dict = {
        "Core": {"Id": "CORE-000598", "Status": "Published", "Version": "1"},
        "Rule_Type": "Dataset Metadata Check",
        "Sensitivity": "Dataset",
        "Executability": "Fully Executable",
        "Authorities": [
            {"Organization": "CDISC", "Standards": [
                {"Name": "SDTMIG", "References": [{
                    "Citations": [{"Cited_Guidance": "test", "Document": "x", "Section": "x"}],
                    "Origin": "test", "Rule_Identifier": {"Id": "x", "Version": "1"},
                    "Version": "2.0"}], "Version": "3.4"}]}
        ],
        "Scope": {"Classes": {"Include": ["ALL"]}, "Domains": {}},
        "Check": {"all": [
            {"name": "dataset_name", "operator": "prefix_not_equal_to",
             "prefix": 2, "value": "DOMAIN"}
        ]},
        "Outcome": {
            "Message": "Dataset name does not begin with DOMAIN value",
            "Output_Variables": ["DOMAIN"],
        },
    }
    rule = Rule.from_cdisc_metadata(rule_dict)

    datasets = [DummyDataset({
        "name": "AE", "filename": "ae.xpt", "label": "Adverse Events",
        "file_size": 1000,
        "records": {"STUDYID": ["S1"], "DOMAIN": ["XY"], "USUBJID": ["001"], "AESEQ": [1]},
    })]
    cache = InMemoryCacheService()
    svc = DummyDataService(cache, MagicMock(), MagicMock(), data=datasets)
    metas = [svc.get_raw_dataset_metadata(dataset_name="ae.xpt")]

    engine = RulesEngine(
        data_service=svc, cache=cache, config_obj=MagicMock(),
        standard="sdtmig", standard_version="3.4",
    )
    results = engine.validate_single_rule(rule, metas)
    issues = [
        e for e in results.get("AE", [])
        if e.get("executionStatus") == "issue reported"
    ]
    assert issues, "CORE-000598 must fire for AE with DOMAIN=XY"
    err = issues[0]["errors"][0]
    assert err["value"]["DOMAIN"] == "XY", (
        f"Expected DOMAIN=XY in the violation value map, got {err['value']!r}; "
        "this means either the DOMAIN column wasn't added to the metadata "
        "frame or the DummyDataService dropped first_record."
    )
