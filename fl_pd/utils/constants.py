import enum

CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}

DATASETS = ("adni", "ppmi", "qpn", "site1", "site2", "site3")  # sorted

COLS_PHENO = ["COG_DECLINE", "AGE", "SEX", "DIAGNOSIS", "IS_CONTROL"]

# cognitive decline thresholds
THRESHOLD_MOCA_RATE = -1
THRESHOLD_MMSE_RATE = -1

PCN_MODEL_INFO_A2009S_ASEG = ("lifespan_57K_82sites", "site_ids_ct_82sites.txt")
PCN_MODEL_INFO_DK = ("lifespan_DK_46K_59sites", "site_ids_ct_dk_59sites.txt")

# directory naming conventions
DNAME_LATEST = "latest"
DATE_FORMAT = "%Y-%m-%d"


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
