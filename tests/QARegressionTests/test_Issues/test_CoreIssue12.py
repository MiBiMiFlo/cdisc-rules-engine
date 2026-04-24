import os
import subprocess
import unittest

import pytest
import json
from conftest import get_python_executable


@pytest.mark.regression
class TestCoreIssue12(unittest.TestCase):
    """Regression for Variable Metadata Check + ``exists`` divergence.

    CORE-000012 is a ``Rule_Type: Variable Metadata Check`` rule whose Check
    is ``{"name": "AEOCCUR", "operator": "exists"}``. SDTMIG prohibits
    AEOCCUR in the AE domain, so its presence should fire one violation.

    Python's engine used to evaluate the Check against the frame returned
    by ``VariablesMetadataDatasetBuilder`` — whose columns are meta-field
    names (``variable_name``, ``variable_label``, …), not the source
    dataset's columns — and the ``exists`` operator would conclude that
    AEOCCUR "doesn't exist" → 0 violations. Java evaluates the same rule
    against the original dataset's column metadata and emits 1 violation.

    Fix (Phase 4 / Finding #9 Option 1): the builder now attaches an
    ``_original_columns`` attribute carrying the source dataset's column
    set; the ``exists`` operator consults it when the bare column lookup
    misses on the metadata projection.

    Rule body is extracted verbatim — no deviations.
    """

    def test_vmc_exists_on_original_columns(self):
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
            os.path.join("tests", "resources", "CoreIssue12", "ae.json"),
            "-lr",
            os.path.join("tests", "resources", "CoreIssue12", "rule.json"),
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
