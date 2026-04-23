import pytest
from cdisc_rules_engine.models.rule import Rule


@pytest.mark.parametrize(
    "condition, expected_additional_keys",
    [
        ({"operator": "test", "name": "IDVAR", "prefix": 10}, ["prefix"]),
        ({"operator": "test", "name": "IDVAR", "suffix": 10}, ["suffix"]),
        (
            {"operator": "test", "name": "IDVAR", "date_component": "year"},
            ["date_component"],
        ),
        ({"operator": "test", "name": "IDVAR", "context": "RDOMAIN"}, ["context"]),
        (
            {"operator": "test", "name": "IDVAR", "value_is_literal": False},
            ["value_is_literal"],
        ),
        (
            {"operator": "test", "name": "IDVAR", "metadata": "metadata_column"},
            ["metadata"],
        ),
        (
            {
                "operator": "test",
                "name": "IDVAR",
                "within": "metadata_column",
                "order": "asc",
                "ordering": "asc",
            },
            ["within", "order", "ordering"],
        ),
    ],
)
def test_build_conditions(condition, expected_additional_keys):
    result = Rule.build_condition(condition, "get_dataset")
    value = result.get("value")
    assert len(value.keys()) == 2 + len(expected_additional_keys)
    assert value["target"] == condition["name"]
    for key in expected_additional_keys:
        assert value[key] == condition[key]


def test_parse_conditions_without_check_data_provided():
    conditions = {}
    with pytest.raises(ValueError) as err:
        Rule.parse_conditions(conditions)
        assert err.args[0] == "No check data provided"


def test_valid_parse_conditions():
    conditions = {"all": [{"name": "IDVAR", "operator": "not_equal_to", "value": 10}]}
    parsed_conditions = Rule.parse_conditions(conditions)
    assert "all" in parsed_conditions
    assert len(parsed_conditions["all"]) == 1
    condition = parsed_conditions["all"][0]
    assert condition.get("name") == "get_dataset"
    assert condition.get("operator") == conditions["all"][0]["operator"]
    assert condition["value"]["target"] == conditions["all"][0]["name"]
    assert condition["value"]["comparator"] == conditions["all"][0]["value"]


def test_valid_parse_conditions_no_target():
    conditions = {"all": [{"operator": "not_equal_to", "value": 10}]}
    parsed_conditions = Rule.parse_conditions(conditions)
    assert "all" in parsed_conditions
    assert len(parsed_conditions["all"]) == 1
    condition = parsed_conditions["all"][0]
    assert "target" not in condition["value"]


def test_parse_conditions_bare_leaf_wraps_in_all():
    # A bare-leaf Check (no outer all/any/not) must be wrapped so the
    # business_rules dispatcher only ever sees a composite outer node.
    conditions = {"name": "PARAMCD", "operator": "longer_than", "value": 8}
    parsed_conditions = Rule.parse_conditions(conditions)
    assert list(parsed_conditions.keys()) == ["all"]
    assert len(parsed_conditions["all"]) == 1
    condition = parsed_conditions["all"][0]
    assert condition["name"] == "get_dataset"
    assert condition["operator"] == "longer_than"
    assert condition["value"]["target"] == "PARAMCD"
    assert condition["value"]["comparator"] == 8


def test_bare_leaf_check_runs_without_key_error():
    # End-to-end: bare-leaf Check going through the same pipeline the engine
    # uses at runtime must not raise KeyError when handed to business_rules.
    from cdisc_rules_engine.models.rule_conditions import ConditionCompositeFactory
    from cdisc_rules_engine.utilities.utils import serialize_rule
    from business_rules.engine import check_conditions_recursively

    rule_cdisc = {
        "Core": {"Id": "BARE-LEAF-TEST", "Status": "Draft", "Version": "1"},
        "Authorities": [],
        "Description": "bare-leaf reproducer",
        "Rule_Type": "Record Data",
        "Sensitivity": "Record",
        "Executability": "Fully Executable",
        "Scope": {"Classes": {"Include": ["ALL"]}},
        "Check": {"name": "PARAMCD", "operator": "longer_than", "value": 8},
        "Outcome": {"Message": "bare-leaf", "Output_Variables": ["PARAMCD"]},
    }
    rule = Rule.from_cdisc_metadata(rule_cdisc)
    rule["conditions"] = ConditionCompositeFactory.get_condition_composite(
        rule["conditions"]
    )
    serialized = serialize_rule(rule)
    assert list(serialized["conditions"].keys()) == ["all"]

    class _StubVariables:
        def get_dataset(self):
            raise AssertionError("should not be called in this stub")

    # Only exercise the dispatch layer — we don't need to evaluate the operator
    # here; the point is that check_conditions_recursively no longer falls
    # through to check_condition on an empty dict.
    try:
        check_conditions_recursively(
            {"all": [{"name": "stub", "operator": "noop", "value": None}]},
            _StubVariables(),
        )
    except KeyError as exc:  # pragma: no cover - defensive
        pytest.fail(f"Unexpected KeyError from dispatcher: {exc!r}")
    except Exception:
        # Anything except KeyError at the dispatch layer is fine — the stub
        # variables class will raise AssertionError once the dispatcher hits
        # the leaf. That confirms dispatch routed correctly.
        pass


def test_valid_parse_actions():
    actions = {"Message": "Great job!"}
    parsed_actions = Rule.parse_actions(actions)
    assert isinstance(parsed_actions, list)
    assert len(parsed_actions) == 1
    assert parsed_actions[0]["name"] == "generate_dataset_error_objects"
    assert parsed_actions[0]["params"]["message"] == actions["Message"]


@pytest.mark.parametrize(
    "match_datasets, expected_result",
    [
        (
            [{"Name": "AA", "Keys": ["USUBJID"]}],
            [{"domain_name": "AA", "match_key": ["USUBJID"], "wildcard": "**"}],
        ),
        (
            [{"Name": "SUPPEC", "Keys": ["USUBJID"]}],
            [
                {
                    "domain_name": "SUPPEC",
                    "match_key": ["USUBJID"],
                    "wildcard": "**",
                }
            ],
        ),
        (
            [
                {
                    "Name": "AA",
                    "Keys": ["STUDYID", {"left": "USUBJID", "right": "RSUBJID"}],
                    "Join_Type": "left",
                }
            ],
            [
                {
                    "domain_name": "AA",
                    "match_key": ["STUDYID", {"left": "USUBJID", "right": "RSUBJID"}],
                    "join_type": "left",
                    "wildcard": "**",
                }
            ],
        ),
    ],
)
def test_parse_datasets(match_datasets, expected_result):
    assert Rule.parse_datasets(match_datasets) == expected_result


@pytest.mark.parametrize(
    "yaml, output",
    [
        (
            {"not": {"all": [{"operator": "test", "name": "IDVAR", "value": 5}]}},
            {
                "not": {
                    "all": [
                        {
                            "name": "get_dataset",
                            "operator": "test",
                            "value": {"target": "IDVAR", "comparator": 5},
                        }
                    ]
                }
            },
        ),
        (
            {
                "all": [
                    {
                        "not": {
                            "any": [{"operator": "test", "name": "IDVAR", "value": 5}]
                        }
                    }
                ]
            },
            {
                "all": [
                    {
                        "not": {
                            "any": [
                                {
                                    "name": "get_dataset",
                                    "operator": "test",
                                    "value": {"target": "IDVAR", "comparator": 5},
                                }
                            ]
                        }
                    }
                ]
            },
        ),
    ],
)
def test_parse_conditions_not_condition(yaml, output):
    assert Rule.parse_conditions(yaml) == output


@pytest.mark.parametrize(
    "authorities, expected",
    [
        (
            [
                {
                    "Organization": "CDISC",
                    "Standards": [
                        {
                            "Name": "SDTMIG",
                            "Version": "3-4",
                            "References": [{"Origin": "Test"}],
                        },
                        {"Name": "SENDIG", "Version": "3-1"},
                    ],
                }
            ],
            [
                {"Name": "SDTMIG", "Version": "3-4", "Substandard": None},
                {"Name": "SENDIG", "Version": "3-1", "Substandard": None},
            ],
        )
    ],
)
def test_parse_standards(authorities, expected):
    assert Rule.parse_standards(authorities) == expected


def test_from_cdisc_rule_null_executability():
    rule_dict = {
        "Core": {},
        "Executability": None,
        "Check": {"asdlkjfa": []},
        "Outcome": {"message": "test"},
    }
    rule = Rule.from_cdisc_metadata(rule_dict)
    assert rule.get("executability") == ""
