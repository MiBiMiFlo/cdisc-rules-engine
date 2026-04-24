import os
import subprocess
import unittest

import pytest
import json
from conftest import get_python_executable


@pytest.mark.regression
class TestCoreIssue726(unittest.TestCase):
    """Regression for ``merge_pivot_supp_dataset`` corrupting the SUPP-merge
    when ``IDVAR`` is itself a natural join key (``USUBJID``).

    CORE-000726 declares ``Match_Datasets: [{Keys: [USUBJID], Name: SUPPDM}]``.
    When SUPPDM's ``IDVAR`` column equals ``"USUBJID"``, the old code:

    - appended ``USUBJID`` to ``common_keys`` a second time (it was already
      in ``static_keys``), leaving duplicate USUBJID entries in the merge
      ``on=`` list; and
    - renamed ``IDVARVAL`` → ``USUBJID`` on the SUPP frame, colliding with
      the real USUBJID column already present there.

    Downstream the merge either raised a pandas error or silently dropped
    matched rows, so the rule produced 0 violations instead of 1 on the
    fixture below.

    Fixture shape:

    - ``dm.json``: 2-row DM (row 1 ETHNIC="HISPANIC OR LATINO" — compliant;
      row 2 ETHNIC="" — compliant by Branch 1 on DM alone).
    - ``suppdm.json``: 2-row SUPPDM with ``IDVAR="USUBJID"`` and QNAM
      "CETHNIC" pivoting to a CETHNIC column per subject — row 1 QVAL
      "HISPANIC OR LATINO" (compliant), row 2 QVAL "ASIAN" (not one of the
      allowed values → violation once the merge lands CETHNIC correctly).

    Post-fix: exactly one violation fires on row 2 via the SUPP-aware
    Check branch.
    """

    def test_idvar_usubjid_supp_merge_fires_violation(self):
        command = [
            f"{get_python_executable()}",
            "-m",
            "core",
            "validate",
            "-s",
            "sdtmig",
            "-v",
            "3-4",
            "-dp",
            os.path.join(
                "tests", "resources", "CoreIssue726", "dm.json"
            ),
            "-dp",
            os.path.join(
                "tests", "resources", "CoreIssue726", "suppdm.json"
            ),
            "-lr",
            os.path.join("tests", "resources", "CoreIssue726", "rule.json"),
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
