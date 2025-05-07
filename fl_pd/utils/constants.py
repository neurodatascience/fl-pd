import enum

CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}


class MlSetup(str, enum.Enum):
    SILO = "silo"
    FEDERATED = "federated"
    MEGA = "mega"


class MlProblem(str, enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class MlFramework(str, enum.Enum):
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"


class MlTarget(str, enum.Enum):
    COG_DECLINE = "cog_decline"
    AGE = "age"
    DIAGNOSIS = "diagnosis"


ML_TARGET_MAP = {
    "decline-age-case-aparc": MlTarget.COG_DECLINE,
    "age-sex-hc-aseg": MlTarget.AGE,
    "age-sex-diag-case-hc-aseg": MlTarget.DIAGNOSIS,
    "age-diag-case-hc-aseg": MlTarget.DIAGNOSIS,
    "age-sex-diag-case-hc-aparc": MlTarget.DIAGNOSIS,
}

ML_PROBLEM_MAP = {
    "decline-age-case-aparc": MlProblem.CLASSIFICATION,
    "age-sex-hc-aseg": MlProblem.REGRESSION,
    "age-sex-diag-case-hc-aseg": MlProblem.CLASSIFICATION,
    "age-diag-case-hc-aseg": MlProblem.CLASSIFICATION,
    "age-sex-diag-case-hc-aparc": MlProblem.CLASSIFICATION,
}
