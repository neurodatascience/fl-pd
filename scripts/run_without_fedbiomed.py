#!/usr/bin/env python

import sys
import warnings
from functools import partial
from pathlib import Path

import click
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fl_pd.base_workflow import (
    BaseWorkflow,
    KnownError,
    DEFAULT_N_ITER_NULL,
    DEFAULT_DATASETS,
    DEFAULT_SETUPS,
)
from fl_pd.federation import (
    average_params,
    get_fitted_params,
    get_initial_params,
    set_params,
)
from fl_pd.io import load_Xy
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    MlFramework,
    MlProblem,
    MlSetup,
    MlTarget,
)

DEFAULT_N_ROUNDS = 1

MODEL_MAP = {
    # Ridge
    MlTarget.AGE: partial(Ridge),
    MlTarget.COG_DECLINE: partial(
        LogisticRegression,
        max_iter=1000,
        class_weight="balanced",
        warm_start=True,
        solver="lbfgs",
    ),
    MlTarget.DIAGNOSIS: partial(
        LogisticRegression,
        max_iter=1000,
        class_weight="balanced",
        warm_start=True,
        solver="lbfgs",
    ),
    MlTarget.MMSE: partial(Ridge),
    # # Lasso
    # MlTarget.AGE: partial(Lasso),
    # MlTarget.COG_DECLINE: partial(
    #     LogisticRegression,
    #     penalty="l1",
    #     max_iter=1000,
    #     class_weight="balanced",
    #     warm_start=True,
    #     solver="liblinear",
    # ),
    # MlTarget.DIAGNOSIS: partial(
    #     LogisticRegression,
    #     penalty="l1",
    #     max_iter=1000,
    #     class_weight="balanced",
    #     warm_start=True,
    #     solver="liblinear",
    # ),
    # MlTarget.MMSE: partial(Lasso),
    # # SGD
    # MlTarget.AGE: partial(SGDRegressor, warm_start=True),
    # MlTarget.COG_DECLINE: partial(
    #     SGDClassifier, class_weight="balanced", warm_start=True
    # ),
    # MlTarget.DIAGNOSIS: partial(
    #     SGDClassifier, class_weight="balanced", warm_start=True
    # ),
    # MlTarget.MMSE: partial(SGDRegressor, warm_start=True),
}

warnings.filterwarnings(action="ignore", message="X has feature names")


class SklearnWorkflow(BaseWorkflow):
    def __init__(self, *args, n_rounds: int = DEFAULT_N_ROUNDS, **kwargs):
        super().__init__(*args, framework=MlFramework.SKLEARN, **kwargs)
        self.n_rounds = n_rounds
        self.model = self._get_model()

    def results_suffix(self) -> str:
        return f"pure_sklearn-{super().results_suffix}"

    def _get_model(self):
        model = MODEL_MAP[self.target](random_state=self.random_state)
        if "standardized" not in self.data_tags and "norm" not in self.data_tags:
            model = make_pipeline(StandardScaler(), model)
        return model

    def _get_train_data(
        self,
        i_split: int,
        null: bool,
        setup: MlSetup,
        dataset: str,
        squeeze: bool = True,
    ):
        if setup != MlSetup.MEGA:
            dataset_str = dataset
        else:
            dataset_str = "_".join(["mega"] + list(sorted(self.datasets)))

        fpath = (
            self.dpath_data
            / f"{dataset_str}-{self.data_tags}"
            / f"{dataset_str}-{self.data_tags}-{i_split}train.tsv"
        )

        X, y = load_Xy(
            fpath,
            target_cols=[self.target_col],
            setup=setup,
            dataset=dataset,
            datasets=self.datasets,
            null=null,
        )
        if squeeze:
            y = y.squeeze(axis=1)

        return X, y

    def train(
        self,
        setup: MlSetup,
        n_features: int,
        n_targets: int,
        i_split: int,
        null: bool,
        train_dataset: str,
    ):

        if setup == MlSetup.FEDERATED:

            initial_params = get_initial_params(
                self.model,
                n_features,
                n_targets,
                classes=[0, 1] if self.problem == MlProblem.CLASSIFICATION else None,
            )
            params = initial_params

            for _ in range(self.n_rounds):
                fitted_params = []
                n_samples = []
                for dataset in self.datasets:
                    X_train, y_train = self._get_train_data(
                        i_split=i_split, null=null, setup=setup, dataset=dataset
                    )

                    model = clone(self.model)
                    set_params(model, initial_params)
                    set_params(model, params)

                    model.fit(X_train, y_train)

                    fitted_params.append(get_fitted_params(model))
                    n_samples.append(X_train.shape[0])

                weights = np.array(n_samples) / np.sum(n_samples)
                params = average_params(fitted_params, weights=weights)

            # final model
            model = clone(self.model)
            set_params(model, initial_params)
            set_params(model, params)

        else:
            X_train, y_train = self._get_train_data(
                i_split=i_split, null=null, setup=setup, dataset=train_dataset
            )

            model = clone(self.model)
            model.fit(X_train, y_train)

        return model


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument(
    "dpath_results",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_RESULTS",
)
@click.option("--tag", "data_tags", required=True)
@click.option(
    "--n-rounds",
    type=click.IntRange(min=1),
    default=DEFAULT_N_ROUNDS,
    help="Number of rounds for federated learning setup",
)
@click.option(
    "--setup",
    "setups",
    type=click.Choice(MlSetup, case_sensitive=False),
    multiple=True,
    default=DEFAULT_SETUPS,
)
@click.option(
    "--dataset",
    "datasets",
    type=str,
    multiple=True,
    default=DEFAULT_DATASETS,
    help="Dataset to use for training/testing.",
)
@click.option("--n-splits", type=click.IntRange(min=1), default=1)
@click.option(
    "--null",
    "n_iter_null",
    type=click.IntRange(min=0),
    default=DEFAULT_N_ITER_NULL,
    help="Number of iterations for null data",
)
@click.option(
    "--random-state",
    type=int,
    envvar="RANDOM_SEED",
    help="Random state for reproducibility",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing results files")
def run_without_fedbiomed(**params):
    workflow = SklearnWorkflow(**params)
    workflow.run()


if __name__ == "__main__":
    try:
        run_without_fedbiomed()
    except KnownError as exception:
        click.echo(click.style(f"ERROR: {exception}", fg="red", bold=True))
        sys.exit(1)
