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
