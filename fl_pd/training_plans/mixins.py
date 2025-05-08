from fedbiomed.common.data import DataManager
from fedbiomed.common.training_plans import BaseTrainingPlan

from fl_pd.io import load_Xy


class TrainingPlanMixin:

    def __init__(self):
        super().__init__()
        self.table_vectorizer_ = None

    def init_dependencies(self):
        deps = [
            "from fl_pd.io import load_Xy",
            "from fl_pd.training_plans.mixins import TrainingPlanMixin",
        ]
        try:
            deps.extend(super().init_dependencies())
        except AttributeError:
            pass

        return deps

    def training_data(self: BaseTrainingPlan):
        model_args = self.model_args()
        target_cols = model_args["target_cols"]
        n_features = model_args["n_targets"]
        n_targets = model_args["n_features"]
        shuffle = model_args.get("shuffle", False)

        X_train, y_train, table_vectorizer = load_Xy(
            self.dataset_path,
            target_cols=target_cols,
            return_vectorizer=True,
        )
        self.table_vectorizer_ = table_vectorizer

        if y_train.shape[1] != n_features:
            raise RuntimeError(
                f"Expected {n_features} output features, got {y_train.shape[1]}"
            )
        if X_train.shape[1] != n_targets:
            raise RuntimeError(
                f"Expected {n_targets} input features"
                f", got {X_train.shape[1]}: {X_train.columns}"
            )

        train_kwargs = {"shuffle": shuffle}

        data_manager = DataManager(
            dataset=X_train.values, target=y_train.values, **train_kwargs
        )

        return data_manager
