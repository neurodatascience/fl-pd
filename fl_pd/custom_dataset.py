import json
import warnings
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Type

import numpy as np
import pandas as pd
from fedbiomed.common.dataset._custom_dataset import CustomDataset
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan
from fedbiomed.common.training_plans import FedSGDClassifier, FedSGDRegressor
from nipoppy import NipoppyDataRetriever
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from skrub import TableVectorizer


def training_plan_factory(target: str) -> Type[BaseTrainingPlan]:
    match target:
        case "fl:cognitive_decline_status":
            return ClassificationTrainingPlan
        case "nb:Age":
            return RegressionTrainingPlan
        case "nb:Diagnosis":
            return ClassificationTrainingPlan
        case _:
            raise ValueError(f"Unknown target: {target}")


class NipoppyDatasetMixin:

    # columns
    COL_PARTICIPANT_ID = "participant_id"
    COL_SESSION_ID = "session_id"
    TERMURL_AGE = "nb:Age"
    TERMURL_SEX = "nb:Sex"
    TERMURL_COG_DECLINE = "fl:cognitive_decline_status"
    TERMURL_COG_DECLINE_AVAILABILITY = "fl:cognitive_decline_availability"
    TERMURL_DIAGNOSIS = "nb:Diagnosis"

    # values
    TERMURL_AVAILABLE = "nb:available"
    TERMURL_UNAVAILABLE = "nb:unavailable"
    TERMURL_MALE = "snomed:248153007"
    TERMURL_FEMALE = "snomed:248152002"
    TERMURL_HEALTHY_CONTROL = "ncit:C94342"

    # for derivatives specs
    FS_NAME = "freesurfer"
    FS_VERSION = "7.3.2"
    FS_STATS_NAME = "fs_stats"
    FS_STATS_VERSION = "0.2.1"
    SUFFIX_APARC = "-aparc.DKTatlas-thickness.tsv"
    SUFFIX_ASEG = "-aseg-volume.tsv"

    @classmethod
    def get_derivatives_spec(cls, suffix: str) -> Tuple[str, str, str]:
        return (
            cls.FS_NAME,
            cls.FS_VERSION,
            f"idp/{cls.FS_STATS_NAME}-{cls.FS_STATS_VERSION}/fs{cls.FS_VERSION}{suffix}",
        )

    @classmethod
    def get_aparc_spec(cls) -> Tuple[str, str, str]:
        return cls.get_derivatives_spec(cls.SUFFIX_APARC)

    @classmethod
    def get_aseg_spec(cls) -> Tuple[str, str, str]:
        return cls.get_derivatives_spec(cls.SUFFIX_ASEG)

    @staticmethod
    def transform_dropna(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis="index", how="any")

    @staticmethod
    def transform_cog_decline(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[
            df[NipoppyDatasetMixin.TERMURL_COG_DECLINE_AVAILABILITY]
            == NipoppyDatasetMixin.TERMURL_AVAILABLE
        ]
        df = df.drop(columns=[NipoppyDatasetMixin.TERMURL_COG_DECLINE_AVAILABILITY])
        return df

    @staticmethod
    def transform_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
        # binarize diagnosis: healthy control (0) vs non-healthy-controls (1)
        # NAs remain NAs
        df[NipoppyDatasetMixin.TERMURL_DIAGNOSIS] = df[
            NipoppyDatasetMixin.TERMURL_DIAGNOSIS
        ].map(
            lambda x: (
                0
                if x == NipoppyDatasetMixin.TERMURL_HEALTHY_CONTROL
                else 1 if not pd.isna(x) else x
            ),
        )
        return df

    @staticmethod
    def transform_select_hc(df: pd.DataFrame) -> pd.DataFrame:
        if NipoppyDatasetMixin.TERMURL_DIAGNOSIS in df.columns:
            df = df.loc[
                df[NipoppyDatasetMixin.TERMURL_DIAGNOSIS]
                == NipoppyDatasetMixin.TERMURL_HEALTHY_CONTROL
            ]
            df = df.drop(columns=[NipoppyDatasetMixin.TERMURL_DIAGNOSIS])
        else:
            warnings.warn("TERMURL_DIAGNOSIS column not found in DataFrame")
        return df

    @staticmethod
    def transform_select_patients(df: pd.DataFrame) -> pd.DataFrame:
        if NipoppyDatasetMixin.TERMURL_DIAGNOSIS in df.columns:
            df = df.loc[
                df[NipoppyDatasetMixin.TERMURL_DIAGNOSIS]
                != NipoppyDatasetMixin.TERMURL_HEALTHY_CONTROL
            ]
            df = df.drop(columns=[NipoppyDatasetMixin.TERMURL_DIAGNOSIS])
        else:
            warnings.warn("TERMURL_DIAGNOSIS column not found in DataFrame")
        return df

    @staticmethod
    def transform_aparc(df: pd.DataFrame):
        df = df.drop(
            columns=[
                "lh_MeanThickness_thickness",
                "rh_MeanThickness_thickness",
                "rh_temporalpole_thickness",
            ],
        )
        return df

    @staticmethod
    def transform_aseg(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(
            columns=[
                "3rd-Ventricle",
                "4th-Ventricle",
                "5th-Ventricle",
                "Brain-Stem",
                "BrainSegVol",
                "BrainSegVol-to-eTIV",
                "BrainSegVolNotVent",
                "CC_Anterior",
                "CC_Central",
                "CC_Mid_Anterior",
                "CC_Mid_Posterior",
                "CC_Posterior",
                "CSF",
                "CerebralWhiteMatterVol",
                "CortexVol",
                "Left-Cerebellum-Cortex",
                "Left-Cerebellum-White-Matter",
                "Left-Inf-Lat-Vent",
                "Left-VentralDC",
                "Left-WM-hypointensities",
                "Left-choroid-plexus",
                "Left-non-WM-hypointensities",
                "Left-vessel",
                "MaskVol",
                "MaskVol-to-eTIV",
                "Optic-Chiasm",
                "Right-Cerebellum-Cortex",
                "Right-Cerebellum-White-Matter",
                "Right-Inf-Lat-Vent",
                "Right-VentralDC",
                "Right-WM-hypointensities",
                "Right-choroid-plexus",
                "Right-non-WM-hypointensities",
                "Right-vessel",
                "SubCortGrayVol",
                "SupraTentorialVol",
                "SupraTentorialVolNotVent",
                "SurfaceHoles",
                "TotalGrayVol",
                "WM-hypointensities",
                "lhCerebralWhiteMatterVol",
                "lhCortexVol",
                "lhSurfaceHoles",
                "non-WM-hypointensities",
                "rhCerebralWhiteMatterVol",
                "rhCortexVol",
                "rhSurfaceHoles",
            ]
        )
        return df

    @staticmethod
    def transform_skrub(df: pd.DataFrame) -> pd.DataFrame:
        specific_transformers = []
        if NipoppyDatasetMixin.TERMURL_SEX in df.columns:
            specific_transformers.append(
                (
                    OneHotEncoder(
                        drop=[NipoppyDatasetMixin.TERMURL_MALE], sparse_output=False
                    ),
                    [NipoppyDatasetMixin.TERMURL_SEX],
                )
            )
        if NipoppyDatasetMixin.TERMURL_COG_DECLINE in df.columns:
            specific_transformers.append(
                (
                    OneHotEncoder(
                        drop=[NipoppyDatasetMixin.TERMURL_UNAVAILABLE],
                        sparse_output=False,
                        feature_name_combiner=lambda x, _: x,
                    ),
                    [NipoppyDatasetMixin.TERMURL_COG_DECLINE],
                )
            )

        table_vectorizer = TableVectorizer(specific_transformers=specific_transformers)
        df = table_vectorizer.fit_transform(df)
        return df

    @classmethod
    def dataset_factory(
        cls,
        target: str,
        i_split: int,
        n_splits: int = 10,
        train=True,
        random_state=None,
        null=False,  # TODO
        fname_stats=None,  # TODO
    ) -> CustomDataset:

        match target:
            case cls.TERMURL_COG_DECLINE:
                phenotypes = [
                    cls.TERMURL_AGE,
                    cls.TERMURL_SEX,
                    cls.TERMURL_DIAGNOSIS,
                    cls.TERMURL_COG_DECLINE,
                    cls.TERMURL_COG_DECLINE_AVAILABILITY,
                ]
                derivatives = [
                    cls.get_aparc_spec(),
                ]
                transforms = [
                    cls.transform_aparc,
                    cls.transform_select_patients,
                    cls.transform_cog_decline,
                    cls.transform_dropna,
                    cls.transform_skrub,
                ]
            case cls.TERMURL_AGE:
                phenotypes = [
                    cls.TERMURL_AGE,
                    cls.TERMURL_SEX,
                    cls.TERMURL_DIAGNOSIS,
                ]
                derivatives = [
                    # cls.get_aparc_spec(),
                    cls.get_aseg_spec(),
                ]
                transforms = [
                    # cls.transform_aparc,
                    cls.transform_aseg,
                    cls.transform_select_hc,
                    cls.transform_dropna,
                    cls.transform_skrub,
                ]
            case cls.TERMURL_DIAGNOSIS:
                phenotypes = [
                    cls.TERMURL_AGE,
                    cls.TERMURL_SEX,
                    cls.TERMURL_DIAGNOSIS,
                ]
                derivatives = [
                    cls.get_aparc_spec(),
                    cls.get_aseg_spec(),
                ]
                transforms = [
                    cls.transform_aparc,
                    cls.transform_aseg,
                    cls.transform_diagnosis,
                    cls.transform_dropna,
                    cls.transform_skrub,
                ]
            case _:
                raise ValueError(f"Unknown target: {target}")

        class NipoppyDataset(CustomDataset):

            @cached_property
            def config(self) -> dict:
                if Path(self.path).is_file():
                    dpath_parent = Path(self.path).parent
                else:
                    dpath_parent = Path(self.path)
                return json.loads((dpath_parent / "global_config.json").read_text())[
                    "CUSTOM"
                ]["FL_PD"]

            def filter_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
                if (session_id := self.config.get("SINGLE_SESSION", None)) is not None:
                    df = df.query(
                        f"{NipoppyDatasetMixin.COL_SESSION_ID} == '{session_id}'"
                    )
                elif (
                    mapping_file_path := self.config.get("MAPPING_FILE", None)
                ) is not None:
                    df_mapping = pd.read_csv(
                        Path(self.path, mapping_file_path),
                        sep="\t",
                        header=None,
                        names=[
                            NipoppyDatasetMixin.COL_PARTICIPANT_ID,
                            NipoppyDatasetMixin.COL_SESSION_ID,
                        ],
                        dtype=str,
                    )
                    idx = pd.MultiIndex.from_frame(df_mapping)
                    df = df.loc[idx]

                # make sure participant IDs are unique
                participant_ids = df.index.get_level_values(
                    NipoppyDatasetMixin.COL_PARTICIPANT_ID
                )
                if len(participant_ids) != len(set(participant_ids)):
                    raise ValueError("Some participants have more than one session")

                return df

            def standardize_df(
                self, df: pd.DataFrame, cols_to_ignore=None
            ) -> pd.DataFrame:
                if cols_to_ignore is None:
                    cols_to_ignore = []
                cols_to_ignore.append("dataset")

                fpath_stats = Path(self.config["STATS"], fname_stats)
                df_stats = pd.read_csv(fpath_stats, sep="\t", index_col=0)
                for col in df.columns:
                    if col in cols_to_ignore:
                        continue
                    if col not in df_stats.columns:
                        raise ValueError(f"{col=} not in {fpath_stats=}")
                    mean = df_stats.at["mean", col]
                    std = df_stats.at["std", col]
                    if std == 0:
                        raise ValueError(f"std is zero for {col=} in {fpath_stats=}")
                    df[col] = (df[col] - mean) / std
                return df

            def read(self):
                self.phenotypes = phenotypes
                self.derivatives = derivatives
                self.transforms = transforms
                self.n_splits = n_splits
                self.i_split = i_split
                self.target = target
                self.random_state = random_state

                need_cv_split = True

                if Path(self.path).is_file():
                    df = pd.read_csv(
                        self.path,
                        sep="\t",
                        dtype={
                            NipoppyDatasetMixin.COL_PARTICIPANT_ID: str,
                            NipoppyDatasetMixin.COL_SESSION_ID: str,
                        },
                    ).set_index(
                        [
                            NipoppyDatasetMixin.COL_PARTICIPANT_ID,
                            NipoppyDatasetMixin.COL_SESSION_ID,
                        ]
                    )
                    df = df.query("i_split == @i_split and train == @train")
                    df = df.drop(columns=["i_split", "train"])
                    need_cv_split = False
                else:
                    retriever = NipoppyDataRetriever(self.path)
                    try:
                        df = retriever.get_tabular_data(
                            phenotypes=phenotypes,
                            derivatives=derivatives,
                        )
                    except KeyError as e:
                        warnings.warn(
                            f"Error retrieving data from {self.path}: {e}. Trying again without diagnosis column."
                        )
                        df = retriever.get_tabular_data(
                            phenotypes=[
                                phenotype
                                for phenotype in phenotypes
                                if phenotype != NipoppyDatasetMixin.TERMURL_DIAGNOSIS
                            ],
                            derivatives=derivatives,
                        )

                    # filter sessions
                    df = self.filter_sessions(df)

                # for building mega dataset
                self.df_before_transforms: pd.DataFrame = df.copy()

                # apply transforms
                for transform in transforms:
                    df = transform(df)
                    if not isinstance(df, pd.DataFrame):
                        raise TypeError(
                            "Transform functions must return a pandas DataFrame."
                        )

                # after transforms but before splits
                self.df_after_transforms: pd.DataFrame = df.copy()

                if need_cv_split:
                    # stratification variable
                    if target == NipoppyDatasetMixin.TERMURL_AGE:
                        bin_size = self.config.get("AGE_BIN_SIZE", 5)
                        bin_min = df[target].min() - 1
                        bin_max = (
                            bin_size * (((df[target].max() - bin_min) // bin_size) + 1)
                            + bin_min
                            + 1
                        )
                        bins = np.arange(bin_min, bin_max, bin_size)
                        y = pd.cut(df[target], bins=bins)
                        # print(y.value_counts(dropna=False))
                        if y.isna().any():
                            raise RuntimeError(
                                "Some samples have NaN values in the stratification variable after binning."
                            )
                        y = y.astype(str)
                    else:
                        y = df[target]

                    # get CV fold
                    cv = StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=random_state
                    )
                    splits = cv.split(np.arange(len(df)), y=y)
                    for _ in range(i_split + 1):
                        idx_train, idx_test = next(splits)
                    if train:
                        idx = df.index[idx_train]
                    else:
                        idx = df.index[idx_test]
                    self.idx = idx
                    df = df.loc[idx]

                if fname_stats is not None:
                    df = self.standardize_df(df, cols_to_ignore=[target])

                self.df: pd.DataFrame = df.copy()
                self.y = self.df[[self.target]]
                self.X = self.df.drop(labels=[self.target], axis="columns")

                if null:
                    rng = np.random.default_rng()
                    idx = np.arange(len(self.y))
                    rng.shuffle(idx)
                    self.y = self.y.iloc[idx]

                return self.X, self.y

            def __len__(self) -> int:
                return len(self.df)

            def get_item(self, idx) -> Tuple[np.ndarray, np.ndarray | None]:
                return self.X.iloc[idx].to_numpy(), self.y.iloc[idx].to_numpy()

        return NipoppyDataset()

    def init_dependencies(self: BaseTrainingPlan) -> List[str]:
        deps = [
            "import json",
            "import warnings",
            "from functools import cached_property",
            "from typing import List, Tuple, Type",
            "import numpy as np",
            "import pandas as pd",
            "from fl_pd.custom_dataset import NipoppyDatasetMixin",
            "from fedbiomed.common.dataset._custom_dataset import CustomDataset",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.training_plans._base_training_plan import BaseTrainingPlan",
            "from fedbiomed.common.training_plans import FedSGDClassifier, FedSGDRegressor",
            "from nipoppy import NipoppyDataRetriever",
            "from sklearn.model_selection import StratifiedKFold",
            "from sklearn.preprocessing import OneHotEncoder",
            "from skrub import TableVectorizer",
        ]
        return deps

    def training_data(self: BaseTrainingPlan) -> DataManager:
        model_args = self.model_args()
        dataset = self.dataset_factory(
            target=model_args["target"],
            i_split=model_args["i_split"],
            n_splits=model_args.get("n_splits", 10),
            random_state=model_args.get("random_state", None),
            null=model_args.get("null", False),
            fname_stats=model_args.get("fname_stats", None),
        )
        return DataManager(dataset=dataset, shuffle=model_args.get("shuffle", False))


class ClassificationTrainingPlan(NipoppyDatasetMixin, FedSGDClassifier):
    pass


class RegressionTrainingPlan(NipoppyDatasetMixin, FedSGDRegressor):
    pass
