from typing import Iterable, List, Set, Union

from cdisc_rules_engine.models.dataset.dataset_interface import DatasetInterface
from cdisc_rules_engine.models.sdtm_dataset_metadata import SDTMDatasetMetadata

from cdisc_rules_engine.enums.join_types import JoinTypes
from cdisc_rules_engine.services import logger
from cdisc_rules_engine.interfaces import (
    CacheServiceInterface,
    DataServiceInterface,
)
from cdisc_rules_engine.models.rule_conditions.condition_composite import (
    ConditionComposite,
)
from cdisc_rules_engine.models.rule_conditions.not_condition_composite import (
    NotConditionComposite,
)
from cdisc_rules_engine.models.rule_conditions.single_condition import (
    SingleCondition,
)
from cdisc_rules_engine.utilities.data_processor import DataProcessor
from cdisc_rules_engine.utilities.rule_processor import RuleProcessor
from cdisc_rules_engine.utilities.utils import (
    replace_pattern_in_list_of_strings,
    get_sided_match_keys,
)
from cdisc_rules_engine.exceptions.custom_exceptions import PreprocessingError
import os
import pandas as pd


class DatasetPreprocessor:
    """
    The class is responsible for preprocessing the dataset
    before starting the validation.

    Preprocessing means merging the dataset with other datasets
    provided in the rule. Depending on the dataset type the merging
    process is different:
        Usual SDTM datasets are merged on match keys provided in the rule.
        Relationship datasets are merged on the columns that describe their relation.

    preprocess method is an entrypoint for the class clients.
    """

    def __init__(
        self,
        dataset: DatasetInterface,
        dataset_metadata: SDTMDatasetMetadata,
        data_service: DataServiceInterface,
        cache_service: CacheServiceInterface,
    ):
        self._dataset: DatasetInterface = dataset
        self._dataset_metadata: SDTMDatasetMetadata = dataset_metadata
        self._data_service = data_service
        self._rule_processor = RuleProcessor(self._data_service, cache_service)
        # Accumulates the set of QNAM values pivoted into columns by
        # ``merge_pivot_supp_dataset`` during this preprocess run. Used by
        # ``_rewrite_qnam_equal_to_leaves`` to fix up Check leaves that
        # reference QNAM pre-pivot — see CORE-000597 / Symptom B / Finding #24.
        self._pivoted_qnams: Set[str] = set()

    def preprocess(  # noqa
        self, rule: dict, datasets: Iterable[SDTMDatasetMetadata]
    ) -> DatasetInterface:
        """
        Preprocesses the dataset by merging it with the
        datasets from the provided rule.
        """
        # Reset per-call state. ``_pivoted_qnams`` accumulates QNAM values
        # observed in any SUPP/SQ child during this preprocess pass; the
        # set drives the post-merge Check-leaf rewrite at the bottom of
        # this method (Symptom B / Finding #24 — see CORE-000597).
        self._pivoted_qnams = set()

        rule_datasets: List[dict] = rule.get("datasets")
        if not rule_datasets:
            return self._dataset  # nothing to preprocess

        rule_targets = self._rule_processor.extract_referenced_variables_from_rule(rule)
        # Get all targets that reference the merge domain.
        result: DatasetInterface = self._dataset.copy()
        merged_domains = set()
        for domain_details in rule_datasets:
            domain_name: str = domain_details.get("domain_name")
            is_child = bool(domain_details.get("child"))
            # download other datasets from blob storage and merge
            if is_child:
                file_infos = []
                # find parent of SUPP or SQAP dataset
                if (
                    (domain_name[:4] == "SUPP" or domain_name[:4] == "SQAP")
                    and self._dataset_metadata.is_supp
                    and self._dataset_metadata.rdomain
                ):
                    if (
                        domain_name == "SUPP--"
                        or domain_name == self._dataset_metadata.name
                    ):
                        file_infos: list[SDTMDatasetMetadata] = [
                            item
                            for item in datasets
                            if (item.domain == self._dataset_metadata.rdomain)
                        ]
                # find parent of other datasets
                elif (
                    domain_name == self._dataset_metadata.domain
                    or domain_name == self._dataset_metadata.name
                ):
                    file_infos: list[SDTMDatasetMetadata] = self._find_parent_dataset(
                        datasets, domain_details
                    )
            else:
                if self._is_split_domain(domain_name):
                    continue
                target_domain_name: str = (
                    self._dataset_metadata.domain or self._dataset_metadata.name
                )
                file_infos: list[SDTMDatasetMetadata] = [
                    item
                    for item in datasets
                    if (
                        item.domain == domain_name
                        or item.name == domain_name
                        or item.unsplit_name == domain_name
                        or (
                            domain_name == "SUPP--"
                            and (not self._dataset_metadata.is_supp)
                            and item.rdomain == target_domain_name
                        )
                    )
                ]

            if not file_infos and not (
                (self._dataset_metadata.is_supp and domain_name == "SUPP--")
                or self._dataset_metadata.name == "RELREC"
            ):
                logger.info(
                    f"Related dataset '{domain_name}' not found for {self._dataset_metadata.name}. "
                    f"Skipping merge for this dataset."
                )
                continue

            for file_info in file_infos:
                if file_info.domain in merged_domains:
                    continue

                # Try to download the dataset
                try:
                    other_dataset: DatasetInterface = self._download_dataset(
                        file_info.data_service_identifier
                    )
                except Exception as e:
                    raise PreprocessingError(
                        f"Failed to download dataset '{file_info.data_service_identifier}' for preprocessing: {str(e)}"
                    )

                referenced_targets = set(
                    [
                        target.replace(f"{domain_name}.", "")
                        for target in rule_targets
                        if isinstance(target, str)
                        and target.startswith(f"{domain_name}.")
                    ]
                )
                other_dataset.columns = [
                    (
                        f"{domain_name}.{column}"
                        if column in referenced_targets
                        else column
                    )
                    for column in other_dataset.columns
                ]
                if is_child:
                    result = self._child_merge_datasets(
                        left_dataset=result,
                        left_dataset_domain_name=self._dataset_metadata.domain,
                        right_dataset=other_dataset,
                        right_dataset_domain_name=file_info.domain,
                        match_keys=domain_details.get("match_key"),
                        datasets=datasets,
                    )
                    merged_domains.add(file_info.domain)
                else:
                    result = self._merge_datasets(
                        left_dataset=result,
                        left_dataset_domain_name=self._dataset_metadata.domain,
                        right_dataset=other_dataset,
                        right_dataset_domain_details=domain_details,
                        datasets=datasets,
                        renamed_right_columns=referenced_targets,
                    )
                    merged_domains.add(
                        file_info.domain if file_info.domain else file_info.name
                    )
        # CORE-000597 / Symptom B / Finding #24: when a SUPP/SQ child has been
        # pivot-merged into ``result``, each unique QNAM has become its own
        # column on the merged frame and the original ``QNAM``/``QVAL``/
        # ``QLABEL`` columns have been dropped. Rules authored with the
        # pre-pivot leaf shape ``{name: "QNAM", operator: "equal_to",
        # value: "AESOSP"}`` (CORE-000597, 000726, 000746, 000747) would
        # otherwise raise ``KeyError`` on column lookup; rewrite them in
        # place to ``{name: "AESOSP", operator: "non_empty"}`` which is the
        # post-pivot semantic equivalent ("a SUPP child has QNAM=AESOSP" =
        # "the AESOSP column is populated for this parent row"). Also
        # synthesize ``QNAM`` and ``QVAL`` columns on the merged frame so
        # sibling QVAL Check leaves and ``Outcome.Output_Variables`` lists
        # referencing QNAM/QVAL resolve to the matched QNAM literal and its
        # corresponding pivoted value.
        if self._pivoted_qnams:
            rewritten_qnams = self._rewrite_qnam_equal_to_leaves(
                rule, self._pivoted_qnams
            )
            if rewritten_qnams:
                self._synthesize_qnam_qval_columns(result, rewritten_qnams)
        return result

    def _rewrite_qnam_equal_to_leaves(
        self, rule: dict, pivoted_qnams: Set[str]
    ) -> Set[str]:
        """Walk ``rule['conditions']`` and rewrite any
        ``{name: "QNAM", operator: "equal_to", value: <X>}`` leaf where
        ``<X>`` is one of the pivoted QNAMs to
        ``{name: <X>, operator: "non_empty"}``. Returns the set of
        comparator (``<X>``) values that were rewritten — used downstream to
        synthesize ``QNAM``/``QVAL`` columns on the merged frame. Mirrors
        Java's leaf-evaluator fallback that resolves unqualified ``QNAM``
        against the joined SUPP dataset; both engines now agree post-pivot.
        Idempotent.
        """
        rewritten: Set[str] = set()
        conditions = rule.get("conditions")
        if conditions is None:
            return rewritten
        self._walk_conditions_for_qnam_rewrite(
            conditions, pivoted_qnams, rewritten
        )
        return rewritten

    def _walk_conditions_for_qnam_rewrite(
        self, condition, pivoted_qnams: Set[str], rewritten: Set[str]
    ) -> None:
        if isinstance(condition, ConditionComposite):
            for cond_list in condition.get_conditions().values():
                for c in cond_list:
                    self._walk_conditions_for_qnam_rewrite(
                        c, pivoted_qnams, rewritten
                    )
            return
        if isinstance(condition, NotConditionComposite):
            # NotConditionComposite wraps a composite; recurse via the
            # private attribute since there's no public accessor for it.
            self._walk_conditions_for_qnam_rewrite(
                condition._condition_composite, pivoted_qnams, rewritten
            )
            return
        if isinstance(condition, SingleCondition):
            leaf = condition.get_conditions()
            if not isinstance(leaf, dict):
                return
            # ``Rule.from_cdisc_metadata`` rewrites
            # ``{name: "QNAM", operator: "equal_to", value: "AESOSP"}`` into the
            # ``business_rules``-flavoured shape
            # ``{name: "get_dataset", operator: "equal_to",
            #     value: {target: "QNAM", comparator: "AESOSP",
            #             value_is_literal: true}}``.
            # Match on that shape: name=get_dataset, operator=equal_to,
            # value.target=QNAM, value.comparator in pivoted_qnams.
            if leaf.get("operator") != "equal_to":
                return
            if leaf.get("name") != "get_dataset":
                return
            value = leaf.get("value")
            if not isinstance(value, dict):
                return
            if value.get("target") != "QNAM":
                return
            comparator = value.get("comparator")
            if not isinstance(comparator, str) or comparator not in pivoted_qnams:
                return
            # Rewrite to the post-pivot equivalent:
            # ``{name: "get_dataset", operator: "non_empty",
            #    value: {target: <comparator>}}`` — i.e. the column named after
            # the (now-pivoted) QNAM is populated for this parent row.
            leaf["operator"] = "non_empty"
            leaf["value"] = {"target": comparator}
            rewritten.add(comparator)

    def _synthesize_qnam_qval_columns(
        self, result: DatasetInterface, rewritten_qnams: Set[str]
    ) -> None:
        """Add ``QNAM`` and ``QVAL`` columns to the merged frame so sibling
        Check leaves that still reference the pre-pivot ``QVAL`` column, and
        ``Outcome.Output_Variables`` entries naming ``QNAM``/``QVAL``,
        resolve to the right values per parent row. For each row, QNAM is
        set to the first ``<X>`` (in ``rewritten_qnams``, sorted for
        determinism) whose pivoted column is populated, and QVAL is set to
        that column's value. Rows where no rewritten QNAM matches get NA on
        both columns. ``process_supp`` already dropped the originals so
        adding them back here doesn't clobber anything.
        """
        try:
            existing_columns = list(result.columns)
        except Exception:
            return
        # Defensive: don't overwrite columns the test author or upstream
        # somehow already set on the frame.
        if "QNAM" in existing_columns or "QVAL" in existing_columns:
            return
        candidates = [x for x in sorted(rewritten_qnams) if x in existing_columns]
        if not candidates:
            return
        qnam_col: List = []
        qval_col: List = []
        try:
            row_count = len(result)
        except Exception:
            return
        for idx in range(row_count):
            matched_qnam = pd.NA
            matched_qval = pd.NA
            for x in candidates:
                v = result[x].iloc[idx]
                if pd.notna(v) and str(v) != "":
                    matched_qnam = x
                    matched_qval = v
                    break
            qnam_col.append(matched_qnam)
            qval_col.append(matched_qval)
        result["QNAM"] = qnam_col
        result["QVAL"] = qval_col

    def _find_parent_dataset(
        self, datasets: Iterable[SDTMDatasetMetadata], domain_details: dict
    ) -> SDTMDatasetMetadata:
        matching_datasets = []
        try:
            if "RDOMAIN" in self._dataset.columns:
                rdomain_column = self._dataset.data["RDOMAIN"]
                unique_domains = set(rdomain_column.unique())
                for dataset in datasets:
                    if dataset.domain in unique_domains:
                        matching_datasets.append(dataset)
            else:
                match_keys = domain_details.get("match_key")
                for dataset in datasets:
                    has_all_match_keys = all(
                        match_key in dataset.first_record for match_key in match_keys
                    )
                    if has_all_match_keys:
                        matching_datasets.append(dataset)
        except Exception as e:
            raise PreprocessingError(
                f"Error during parent dataset search. "
                f"Current dataset: {self._dataset_metadata.name}, "
                f"Match keys: {domain_details.get('match_key')}, "
                f"Error: {str(e)}"
            )
        if not matching_datasets:
            logger.warning(
                f"Child specified in match but no parent datasets found for: {domain_details}"
            )
        return matching_datasets

    def _is_split_domain(self, domain: str) -> bool:
        return domain == self._dataset_metadata.unsplit_name

    def _download_dataset(self, filename: str) -> DatasetInterface:
        return self._data_service.get_dataset(
            dataset_name=os.path.join(
                os.path.dirname(self._dataset_metadata.full_path), filename
            )
        )

    def _child_merge_datasets(
        self,
        left_dataset: DatasetInterface,
        left_dataset_domain_name: str,
        right_dataset: DatasetInterface,
        right_dataset_domain_name: str,
        match_keys: List[str],
        datasets: Iterable[SDTMDatasetMetadata] = None,
    ) -> DatasetInterface:
        is_supplemental, rdomain_dataset = self._classify_dataset(
            left_dataset, self._dataset_metadata
        )
        has_idvar_keys = "IDVAR" in match_keys or "IDVARVAL" in match_keys
        if (is_supplemental or rdomain_dataset) and has_idvar_keys:
            return self._merge_rdomain_dataset(
                left_dataset,
                left_dataset_domain_name,
                right_dataset,
                right_dataset_domain_name,
                match_keys,
            )
        else:
            left_dataset_match_keys = replace_pattern_in_list_of_strings(
                get_sided_match_keys(match_keys=match_keys, side="left"),
                "--",
                left_dataset_domain_name,
            )
            right_dataset_match_keys = replace_pattern_in_list_of_strings(
                get_sided_match_keys(match_keys=match_keys, side="right"),
                "--",
                right_dataset_domain_name,
            )
            try:
                result = left_dataset.merge(
                    right_dataset.data,
                    how="left",
                    left_on=left_dataset_match_keys,
                    right_on=right_dataset_match_keys,
                    suffixes=("", f".{right_dataset_domain_name}"),
                )
                return result
            except Exception as e:
                raise PreprocessingError(
                    f"Merge operation failed during child merge. "
                    f"Left dataset: {left_dataset_domain_name}, "
                    f"Right dataset: {right_dataset_domain_name}, "
                    f"Error: {str(e)}"
                )

    def _classify_dataset(
        self, dataset: DatasetInterface, metadata: SDTMDatasetMetadata
    ) -> tuple[bool, bool]:
        is_supplemental = metadata and metadata.is_supp
        has_rdomain_not_supp = "RDOMAIN" in dataset.columns and not is_supplemental

        return is_supplemental, has_rdomain_not_supp

    def _merge_rdomain_dataset(
        self,
        left_dataset: DatasetInterface,
        left_dataset_domain_name: str,
        right_dataset: DatasetInterface,
        right_dataset_domain_name: str,
        match_keys: List[str],
    ) -> DatasetInterface:
        try:
            relevant_child_records = self._get_relevant_child_records(
                left_dataset, right_dataset_domain_name
            )
            merged_records = self._merge_with_idvar_logic(
                relevant_child_records,
                right_dataset,
                left_dataset_domain_name,
                right_dataset_domain_name,
                match_keys,
            )
            return self._update_dataset_with_merged_records(
                left_dataset, relevant_child_records, merged_records
            )
        except Exception as e:
            raise PreprocessingError(
                f"Failed to merge with IDVAR logic. "
                f"Left dataset: {left_dataset_domain_name}, "
                f"Right dataset: {right_dataset_domain_name}, "
                f"Match keys: {match_keys}, "
                f"Error: {str(e)}"
            )

    def _get_relevant_child_records(
        self, left_dataset: DatasetInterface, parent_domain: str
    ) -> DatasetInterface:
        if self._dataset_metadata.is_supp and self._dataset_metadata.rdomain:
            return left_dataset
        if "RDOMAIN" in left_dataset.columns:
            filtered_records = left_dataset[left_dataset["RDOMAIN"] == parent_domain]
            return filtered_records
        return left_dataset

    def _merge_with_idvar_logic(
        self,
        child_records: DatasetInterface,
        parent_dataset: DatasetInterface,
        left_domain: str,
        right_domain: str,
        match_keys: List[str],
    ) -> pd.DataFrame:
        standard_keys = [key for key in match_keys if key not in ["IDVAR", "IDVARVAL"]]
        has_idvar_keys = "IDVAR" in match_keys or "IDVARVAL" in match_keys
        if standard_keys and has_idvar_keys:
            return self._merge_standard_then_filter_idvar(
                child_records, parent_dataset, left_domain, right_domain, standard_keys
            )
        return self._merge_idvar_only(child_records, parent_dataset, right_domain)

    def _merge_standard_then_filter_idvar(
        self,
        child_records: DatasetInterface,
        parent_dataset: DatasetInterface,
        left_domain: str,
        right_domain: str,
        standard_keys: List[str],
    ) -> pd.DataFrame:
        results = []
        for child_idx, child_row in child_records.iterrows():
            idvar = child_row.get("IDVAR")
            idvarval = child_row.get("IDVARVAL")
            parent_candidates = self._filter_parents_by_standard_keys(
                parent_dataset, child_row, standard_keys, left_domain, right_domain
            )
            final_match = None
            if not parent_candidates.empty and pd.notna(idvar) and pd.notna(idvarval):
                final_match = self._find_idvar_match_in_candidates(
                    parent_candidates, idvar, idvarval, right_domain
                )
            elif not parent_candidates.empty and (pd.isna(idvar) or pd.isna(idvarval)):
                final_match = parent_candidates.iloc[0]
            if final_match is not None:
                merged_record = {**child_row.to_dict(), **final_match.to_dict()}
            else:
                merged_record = child_row.to_dict()
                for col in parent_dataset.columns:
                    if col not in merged_record:
                        merged_record[col] = None
            results.append(merged_record)
        return child_records.__class__.from_records(results)

    def _filter_parents_by_standard_keys(
        self,
        parent_dataset: DatasetInterface,
        child_row: pd.Series,
        standard_keys: List[str],
        left_domain: str,
        right_domain: str,
    ) -> pd.DataFrame:
        left_keys = replace_pattern_in_list_of_strings(
            get_sided_match_keys(standard_keys, "left"), "--", left_domain
        )
        right_keys = replace_pattern_in_list_of_strings(
            get_sided_match_keys(standard_keys, "right"), "--", right_domain
        )
        filter_conditions = pd.Series(
            [True] * len(parent_dataset), index=parent_dataset.index
        )
        for left_key, right_key in zip(left_keys, right_keys):
            if left_key in child_row and right_key in parent_dataset.columns:
                child_value = child_row[left_key]
                if pd.notna(child_value):
                    key_condition = parent_dataset[right_key] == child_value
                    filter_conditions &= key_condition
        filtered_parents = parent_dataset[filter_conditions]
        return filtered_parents

    def _find_idvar_match_in_candidates(  # noqa
        self,
        parent_candidates: pd.DataFrame,
        idvar: str,
        idvarval: str,
        right_domain: str,
    ) -> pd.Series:
        idvar_columns = [
            idvar,
            f"{idvar}.{right_domain}",
            f"{right_domain}.{idvar}",
        ]
        for idvar_col in idvar_columns:
            if idvar_col not in parent_candidates.columns:
                continue
            parent_col = parent_candidates[idvar_col]
            try:
                if parent_col.dtype.kind == "O":
                    for idx in parent_candidates.index:
                        parent_val = parent_candidates.loc[idx, idvar_col]
                        if pd.isna(parent_val):
                            continue
                        converted_val = self._convert_idvarval_for_comparison(
                            idvarval, type(parent_val)
                        )
                        if converted_val == parent_val:
                            return parent_candidates.loc[idx]
                else:
                    converted_idvarval = self._convert_idvarval_for_comparison(
                        idvarval, parent_col.dtype
                    )
                    matching_mask = parent_col == converted_idvarval
                    matching_rows = parent_candidates[matching_mask]
                    if not matching_rows.empty:
                        return matching_rows.iloc[0]
            except Exception as e:
                logger.warning(f"IDVAR matching failed for column '{idvar_col}': {e}")
                continue
            # fallback on string comparison
            try:
                string_parent_col = parent_col.astype(str)
                string_idvarval = str(idvarval).strip()
                string_matching_mask = string_parent_col == string_idvarval
                string_matching_rows = parent_candidates[string_matching_mask]
                if not string_matching_rows.empty:
                    return string_matching_rows.iloc[0]
            except Exception as e:
                logger.warning(f"String fallback failed for column '{idvar_col}': {e}")
        logger.error(f"No IDVAR match found for {idvar}={idvarval}")
        return None

    def _convert_idvarval_for_comparison(
        self, idvarval: str, target_type_or_dtype
    ) -> Union[str, int, float]:  # noqa
        # typematch between the two columns
        try:
            # Python native types
            if target_type_or_dtype in [int, float, str]:
                if target_type_or_dtype == int:
                    return int(float(str(idvarval).strip()))
                elif target_type_or_dtype == float:
                    return float(str(idvarval).strip())
                else:  # str
                    return str(idvarval).strip()
            # Handle pandas inferring object type
            elif hasattr(target_type_or_dtype, "__name__"):
                type_name = target_type_or_dtype.__name__
                if "int" in type_name.lower():
                    return int(float(str(idvarval).strip()))
                elif "float" in type_name.lower():
                    return float(str(idvarval).strip())
                else:
                    return str(idvarval).strip()
            # fallback on string type
            return str(idvarval).strip()
        except Exception as e:
            logger.warning(
                f"Type conversion failed for '{idvarval}' -> {target_type_or_dtype}: {e}"
            )
            return str(idvarval).strip()

    def _merge_idvar_only(
        self,
        child_records: DatasetInterface,
        parent_dataset: DatasetInterface,
        right_domain: str,
    ) -> pd.DataFrame:
        results = []
        for child_idx, child_row in child_records.iterrows():
            idvar = child_row.get("IDVAR")
            idvarval = child_row.get("IDVARVAL")
            parent_match = None
            if pd.notna(idvar) and pd.notna(idvarval):
                parent_match = self._find_idvar_match_in_candidates(
                    parent_dataset.data, idvar, idvarval, right_domain
                )
            if parent_match is not None:
                merged_record = {**child_row.to_dict(), **parent_match.to_dict()}
            else:
                merged_record = child_row.to_dict()
                for col in parent_dataset.columns:
                    if col not in merged_record:
                        merged_record[col] = None
            results.append(merged_record)
        return child_records.__class__.from_records(results)

    def _update_dataset_with_merged_records(
        self,
        original_dataset: DatasetInterface,
        old_records: DatasetInterface,
        merged_records: pd.DataFrame,
    ) -> DatasetInterface:
        if old_records.empty:
            return original_dataset
        if self._dataset_metadata.is_supp:
            return self._dataset.__class__(data=merged_records)
        remaining_records = original_dataset.data.drop(old_records.index)
        updated_data = pd.concat([remaining_records, merged_records], ignore_index=True)
        return self._dataset.__class__(data=updated_data)

    def _merge_datasets(  # noqa
        self,
        left_dataset: DatasetInterface,
        left_dataset_domain_name: str,
        right_dataset: DatasetInterface,
        right_dataset_domain_details: dict,
        datasets: List[dict],
        renamed_right_columns: Set[str] = None,
    ) -> DatasetInterface:
        """
        Merges datasets on their match keys.
        Identifies dataset type and merges based on it.
        """
        # replace -- pattern in match keys for each domain
        match_keys: List[Union[str, dict]] = right_dataset_domain_details.get(
            "match_key"
        )
        left_dataset_match_keys = replace_pattern_in_list_of_strings(
            get_sided_match_keys(match_keys=match_keys, side="left"),
            "--",
            left_dataset_domain_name,
        )
        right_dataset_domain_name: str = right_dataset_domain_details.get("domain_name")
        right_dataset_match_keys = replace_pattern_in_list_of_strings(
            get_sided_match_keys(match_keys=match_keys, side="right"),
            "--",
            right_dataset_domain_name,
        )
        # If `preprocess` renamed right-side columns (e.g. USUBJID → DM.USUBJID
        # because the Check references DM.USUBJID), the match keys must reflect
        # that rename too — otherwise pandas tries to join on a column that no
        # longer exists on the right frame.
        if renamed_right_columns:
            right_dataset_match_keys = [
                f"{right_dataset_domain_name}.{k}"
                if k in renamed_right_columns
                else k
                for k in right_dataset_match_keys
            ]

        # merge datasets based on their type
        if right_dataset_domain_name == "RELREC":
            try:
                result: DatasetInterface = DataProcessor.merge_relrec_datasets(
                    left_dataset=left_dataset,
                    left_dataset_domain_name=left_dataset_domain_name,
                    relrec_dataset=right_dataset,
                    datasets=datasets,
                    dataset_preprocessor=self,
                    wildcard=right_dataset_domain_details.get("wildcard"),
                )
            except Exception as e:
                raise PreprocessingError(
                    f"Failed to merge RELREC dataset in preprocessing. "
                    f"Left dataset: {left_dataset_domain_name}, "
                    f"Wildcard: {right_dataset_domain_details.get('wildcard')}, "
                    f"Match keys: {match_keys}, "
                    f"Error: {str(e)}"
                )
        elif right_dataset_domain_name.startswith(
            "SUPP"
        ) or right_dataset_domain_name.startswith("SQ"):
            # Capture the QNAM values about to be pivoted into columns so the
            # rule's Check leaves can be rewritten post-merge (CORE-000597 /
            # Symptom B / Finding #24). ``merge_pivot_supp_dataset`` ->
            # ``process_supp`` drops the original ``QNAM`` / ``QVAL`` /
            # ``QLABEL`` columns, so any Check authored against the pre-pivot
            # QNAM column would otherwise raise KeyError.
            if "QNAM" in right_dataset.columns:
                try:
                    qnams = right_dataset["QNAM"].unique()
                    self._pivoted_qnams.update(
                        str(q) for q in qnams if pd.notna(q)
                    )
                except Exception:
                    # Tracking pivoted QNAMs is best-effort; if the inspection
                    # fails for any reason the rewrite simply doesn't fire and
                    # the rule preserves its original (possibly broken) shape.
                    pass
            try:
                result: DatasetInterface = DataProcessor.merge_pivot_supp_dataset(
                    dataset_implementation=self._data_service.dataset_implementation,
                    left_dataset=left_dataset,
                    right_dataset=right_dataset,
                )
            except Exception as e:
                raise PreprocessingError(
                    f"Failed to merge supplemental/qualifier dataset. "
                    f"Left dataset: {left_dataset_domain_name} ({len(left_dataset)} rows), "
                    f"SUPP/SQ dataset: {right_dataset_domain_name} ({len(right_dataset)} rows), "
                    f"Error: {str(e)}"
                )
        else:
            try:
                result: DatasetInterface = DataProcessor.merge_sdtm_datasets(
                    left_dataset=left_dataset,
                    right_dataset=right_dataset,
                    left_dataset_match_keys=left_dataset_match_keys,
                    right_dataset_match_keys=right_dataset_match_keys,
                    right_dataset_domain_name=right_dataset_domain_name,
                    join_type=JoinTypes(
                        right_dataset_domain_details.get("join_type", "inner")
                    ),
                )
            except KeyError as e:
                # Handle with COLUMN_NOT_FOUND_IN_DATA in rules_engine
                raise e
            except Exception as e:
                raise PreprocessingError(
                    f"Failed to merge datasets. "
                    f"Left dataset: {left_dataset_domain_name}, Right dataset: {right_dataset_domain_name}, "
                    f"Error: {str(e)}"
                )
        return result
