import enum

CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}

DATASETS = ("adni", "ppmi", "qpn", "site1", "site2", "site3")  # sorted


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
    MMSE = "mmse"
