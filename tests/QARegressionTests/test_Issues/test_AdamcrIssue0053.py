import os
import subprocess
import unittest

import pytest
import json
from conftest import get_python_executable


@pytest.mark.regression
class TestAdamcrIssue0053(unittest.TestCase):
    """Regression for ``shares_no_elements_with`` crashing on suffixed USUBJID
    after a ``dataset_names`` Operation + Match_Datasets merge.

    ADAMCR-0053 declares ``Match_Datasets: [{Keys: [USUBJID], Name: DM}]``
    plus ``Operations: [{id: $datasets, operator: dataset_names}]``, and its
    Check uses a ``not shares_no_elements_with($datasets, [DM])`` leaf
    followed by a ``DM.USUBJID empty`` leaf. After the DM merge, USUBJID
    is renamed with suffix on the ADLB-side joined frame, so the operator's
    ``row[target]`` lookup on the bare ``USUBJID`` name raises
    ``KeyError: 'USUBJID'``. Java resolves the dotted form cleanly.

    Fixture shape mirrors the Cumba spec-runner oracle
    (``spec-runner/specs/ADAMCR-0053.yaml``):

    - ``adlb.json``: 2-row ADLB (STUDYID, USUBJID). Row 1 USUBJID="001"
      exists in DM → no violation; row 2 USUBJID="002" missing from DM →
      DM.USUBJID is null post-merge → violation.
    - ``dm.json``: 1-row DM (STUDYID, DOMAIN, USUBJID="001").

    Deviations from the upstream rule body (Check body essence identical):

    - ``Core.Status`` flipped from ``Published`` to ``Draft`` so the rule
      bypasses ``rule_matches_standard_version`` under ``-s adam``.
    - The bare-leaf ``shares_no_elements_with`` inside the outer ``not``
      wrapped in a single-element ``all``. Logically identical; necessary
      because this branch (off ``upstream/main``) does not carry Phase 1's
      ``Rule.parse_conditions`` fix, which is what prevents the inner bare
      leaf from being flattened to ``{}`` and crashing business_rules with
      ``KeyError: 'name'`` before the preprocessor merge ever runs. Once
      Phase 1 lands upstream, this wrap is a no-op and the fixture could
      be de-wrapped to match the upstream rule verbatim.
    - ``Match_Datasets[0].Join_Type`` added and set to ``"left"``. The rule
      author relied on left-join semantics (find ADLB rows whose USUBJID
      is *missing* in DM), but the upstream rule omits ``Join_Type`` and
      the Python engine defaults to ``inner``, which silently drops the
      unmatched row before the Check runs. Adding the explicit ``"left"``
      matches the Cumba Java spec-runner oracle's expected output.
    """

    def test_shares_no_elements_with_post_merge_suffix(self):
        command = [
            f"{get_python_executable()}",
            "-m",
            "core",
            "validate",
            "-s",
            "adam",
            "-v",
            "adamig-1-3",
            "-dp",
            os.path.join(
                "tests", "resources", "AdamcrIssue0053", "adlb.json"
            ),
            "-dp",
            os.path.join(
                "tests", "resources", "AdamcrIssue0053", "dm.json"
            ),
            "-lr",
            os.path.join("tests", "resources", "AdamcrIssue0053", "rule.json"),
            "-ps",
            "1",
            "-of",
            "json",
        ]
        subprocess.run(command, check=True)

        files = os.listdir()
        json_files = [
            file
            for file in files
            if file.startswith("CORE-Report-") and file.endswith(".json")
        ]
        assert json_files, "Expected a CORE-Report-*.json to be produced"
        json_report_path = sorted(json_files)[-1]
        with open(json_report_path) as f:
            json_report = json.load(f)

        try:
            assert {
                "Conformance_Details",
                "Dataset_Details",
                "Issue_Summary",
                "Issue_Details",
                "Rules_Report",
            }.issubset(json_report.keys())
            assert json_report["Issue_Summary"][0]["issues"] == 1
            assert (
                json_report["Rules_Report"][0]["status"] == "ISSUE REPORTED"
            )
        finally:
            if os.path.exists(json_report_path):
                os.remove(json_report_path)
