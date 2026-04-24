import os
import subprocess
import unittest

import pytest
import json
from conftest import get_python_executable


@pytest.mark.regression
class TestAdamcrIssue0143(unittest.TestCase):
    """Regression for bare-leaf ``Check`` crashing business_rules_enhanced.

    ADAMCR-0143's ``Check`` is a bare leaf with no outer ``all``/``any``/``not``
    wrapper::

        {"name": "PARAMCD", "operator": "longer_than", "value": 8}

    ``Rule.parse_conditions`` used to strip that to ``{}`` (it only reads
    ``all``/``any``/``not`` keys), and ``business_rules.engine.run`` would then
    fall through ``check_conditions_recursively({})`` to ``check_condition({})``
    and crash with ``KeyError: 'name'`` at ``business_rules/engine.py:72``.

    This test runs the actual CLI against the rule lifted verbatim from
    ``rules-adamig-1-3.json`` (with two minimal deviations documented below)
    plus a 2-row ADLBC dataset, and asserts the rule now fires exactly one
    violation.

    Deviations from the upstream rule body (Check itself is verbatim):

    - ``Core.Status`` flipped ``Published`` → ``Draft`` so the rule
      bypasses ``rule_matches_standard_version``, which otherwise rejects
      ``Name=ADaMIG`` rules under ``-s adam``.
    - ``Scope.Classes.Include`` reduced to ``["ALL"]`` so the rule
      bypasses the class-detection scope filter. ADaMIG's library metadata
      does not ship dataset-name→class mappings, so ``ADLBC`` would be
      classified as ``None`` and the rule would be filtered out before
      the bare-leaf ``Check`` ever reached the operator layer.
    """

    def test_bare_leaf_check_fires_violation(self):
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
                "tests",
                "resources",
                "AdamcrIssue0143",
                "dataset.json",
            ),
            "-lr",
            os.path.join("tests", "resources", "AdamcrIssue0143", "rule.json"),
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
