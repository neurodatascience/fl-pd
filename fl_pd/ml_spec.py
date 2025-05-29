from fl_pd.utils.constants import DATASETS, MlProblem, MlTarget

TAG_TO_ML_TARGET_MAP = {
    "decline-age-case-aparc": MlTarget.COG_DECLINE,
    "age-sex-hc-aseg": MlTarget.AGE,
    "age-sex-hc-aseg-55": MlTarget.AGE,
    "age-sex-diag-case-hc-aseg": MlTarget.DIAGNOSIS,
    "age-diag-case-hc-aseg": MlTarget.DIAGNOSIS,
    "age-sex-diag-case-hc-aparc": MlTarget.DIAGNOSIS,
    "age-sex-diag-case-hc-aparc-aseg": MlTarget.DIAGNOSIS,
    "simulated": MlTarget.MMSE,
}


ML_TARGET_TO_PROBLEM_MAP = {
    MlTarget.MMSE: MlProblem.REGRESSION,
    MlTarget.DIAGNOSIS: MlProblem.CLASSIFICATION,
    MlTarget.AGE: MlProblem.REGRESSION,
    MlTarget.COG_DECLINE: MlProblem.CLASSIFICATION,
}


def get_target_from_tag(tag) -> MlTarget:
    for dataset in ("mega",) + DATASETS:
        tag = tag.removeprefix(f"{dataset}-")
    tag = tag.removesuffix("-standardized")
    for known_tag, target in TAG_TO_ML_TARGET_MAP.items():
        if tag.startswith(known_tag):
            return target
    raise ValueError(f"Unable to get ML spec from tag: {tag}")
