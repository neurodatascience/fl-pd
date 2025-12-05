import json
from pathlib import Path
from typing import List, Tuple

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


def training_plan_factory(target: str) -> BaseTrainingPlan:
    match target:
        case "fl:cognitive_decline_status":
            training_plan_class = FedSGDClassifier
        case "nb:Age":
            training_plan_class = FedSGDRegressor
        case "nb:Diagnosis":
            training_plan_class = FedSGDClassifier
        case _:
            raise ValueError(f"Unknown target: {target}")

    class TrainingPlan(training_plan_class):

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

        # for derivatives specs
        FS_NAME = "freesurfer"
        FS_VERSION = "7.3.2"
        FS_STATS_NAME = "fs_stats"
        FS_STATS_VERSION = "0.2.*"
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
                df[TrainingPlan.TERMURL_COG_DECLINE_AVAILABILITY]
                == TrainingPlan.TERMURL_AVAILABLE
            ]
            df = df.drop(columns=[TrainingPlan.TERMURL_COG_DECLINE_AVAILABILITY])
            return df

        def transform_aparc(df: pd.DataFrame):
            df = df.drop(
                columns=[
                    "lh_MeanThickness_thickness",
                    "rh_MeanThickness_thickness",
                    "rh_temporalpole_thickness",
                ],
            )
            return df

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
            if TrainingPlan.TERMURL_SEX in df.columns:
                specific_transformers.append(
                    (
                        OneHotEncoder(
                            drop=[TrainingPlan.TERMURL_MALE], sparse_output=False
                        ),
                        [TrainingPlan.TERMURL_SEX],
                    )
                )
            if TrainingPlan.TERMURL_COG_DECLINE in df.columns:
                specific_transformers.append(
                    (
                        OneHotEncoder(
                            drop=[TrainingPlan.TERMURL_UNAVAILABLE],
                            sparse_output=False,
                            feature_name_combiner=lambda x, _: x,
                        ),
                        [TrainingPlan.TERMURL_COG_DECLINE],
                    )
                )

            table_vectorizer = TableVectorizer(
                specific_transformers=specific_transformers
            )
            df = table_vectorizer.fit_transform(df)
            return df

        @classmethod
        def dataset_factory(
            cls,
            target: str,
            i_split: int,
            n_splits: int = 10,
            train=True,
            rng_seed=None,
        ) -> CustomDataset:

            match target:
                case cls.TERMURL_COG_DECLINE:
                    phenotypes = [
                        cls.TERMURL_AGE,
                        cls.TERMURL_SEX,
                        cls.TERMURL_COG_DECLINE,
                        cls.TERMURL_COG_DECLINE_AVAILABILITY,
                    ]
                    derivatives = [cls.get_aparc_spec()]
                    transforms = [
                        cls.transform_aparc,
                        cls.transform_cog_decline,
                        cls.transform_dropna,
                        cls.transform_skrub,
                    ]
                case cls.TERMURL_AGE:
                    phenotypes = [cls.TERMURL_AGE, cls.TERMURL_SEX]
                    derivatives = [cls.get_aparc_spec(), cls.get_aseg_spec()]
                    transforms = [
                        cls.transform_aparc,
                        cls.transform_aseg,
                        cls.transform_dropna,
                        cls.transform_skrub,
                    ]
                case cls.TERMURL_DIAGNOSIS:
                    phenotypes = [
                        cls.TERMURL_AGE,
                        cls.TERMURL_SEX,
                        cls.TERMURL_DIAGNOSIS,
                    ]
                    derivatives = [cls.get_aparc_spec(), cls.get_aseg_spec()]
                    transforms = [
                        cls.transform_aparc,
                        cls.transform_aseg,
                        cls.transform_dropna,
                        cls.transform_skrub,
                    ]
                case _:
                    raise ValueError(f"Unknown target: {target}")

            class NipoppyDataset(CustomDataset):

                def filter_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
                    config: dict = json.loads(
                        Path(self.path, "global_config.json").read_text()
                    )["CUSTOM"]["FL_PD"]
                    if (session_id := config.get("SINGLE_SESSION", None)) is not None:
                        df = df.query(
                            f"{TrainingPlan.COL_SESSION_ID} == '{session_id}'"
                        )
                    elif (
                        mapping_file_path := config.get("MAPPING_FILE", None)
                    ) is not None:
                        df_mapping = pd.read_csv(
                            Path(self.path, mapping_file_path),
                            sep="\t",
                            header=None,
                            names=[
                                TrainingPlan.COL_PARTICIPANT_ID,
                                TrainingPlan.COL_SESSION_ID,
                            ],
                            dtype=str,
                        )
                        idx = pd.MultiIndex.from_frame(df_mapping)
                        df = df.loc[idx]

                    # make sure participant IDs are unique
                    participant_ids = df.index.get_level_values(
                        TrainingPlan.COL_PARTICIPANT_ID
                    )
                    if len(participant_ids) != len(set(participant_ids)):
                        raise ValueError("Some participants have more than one session")

                    return df

                def read(self):
                    self.phenotypes = phenotypes
                    self.derivatives = derivatives
                    self.transforms = transforms
                    self.n_splits = n_splits
                    self.i_split = i_split
                    self.target = target
                    self.rng_seed = rng_seed

                    retriever = NipoppyDataRetriever(self.path)
                    df = retriever.get_tabular_data(
                        phenotypes=phenotypes,
                        derivatives=derivatives,
                    )

                    # filter sessions
                    df = self.filter_sessions(df)

                    # apply transforms
                    for transform in transforms:
                        df = transform(df)
                        if not isinstance(df, pd.DataFrame):
                            raise TypeError(
                                "Transform functions must return a pandas DataFrame."
                            )

                    # stratification variable
                    if target == TrainingPlan.TERMURL_AGE:
                        bins = np.arange(0, 100, 5)
                        y = pd.cut(df[target], bins=bins).astype(str)
                    else:
                        y = df[target]

                    # get CV fold
                    cv = StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=rng_seed
                    )
                    splits = cv.split(np.arange(len(df)), y=y)
                    for _ in range(i_split + 1):
                        idx_train, idx_test = next(splits)
                    if train:
                        idx = idx_train
                    else:
                        idx = idx_test
                    df = df.iloc[idx]

                    self.df: pd.DataFrame = df.copy()

                def __len__(self) -> int:
                    return len(self.df)

                def get_item(self, idx) -> Tuple[np.ndarray, np.ndarray | None]:
                    row: pd.Series = self.df.iloc[idx]

                    target = row[self.target]
                    data = row.drop(labels=[self.target]).to_numpy()

                    return data, target

            return NipoppyDataset()

        def init_dependencies(self) -> List[str]:
            deps = [
                "import json",
                "from typing import List, Tuple",
                "import numpy as np",
                "import pandas as pd",
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

        def training_data(self):
            model_args = self.model_args()
            dataset = self.dataset_factory(
                target=model_args["target"],
                i_split=model_args["i_split"],
                n_splits=model_args.get("n_splits", 10),
                rng_seed=model_args.get("rng_seed", None),
            )()
            return DataManager(dataset=dataset)

    return TrainingPlan
