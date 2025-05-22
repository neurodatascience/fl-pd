#!/usr/bin/env python

import datetime
import json
import sys
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Generator, Iterable, Optional

import click
from matplotlib.pylab import f
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fl_pd.federation import (
    average_params,
    get_fitted_params,
    get_initial_params,
    set_params,
)
from fl_pd.io import load_Xy
from fl_pd.metrics import get_metrics_map
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    MlProblem,
    MlSetup,
    MlTarget,
)
from fl_pd.ml_spec import ML_TARGET_TO_PROBLEM_MAP, get_target_from_tag

DEFAULT_N_ROUNDS = 1
DEFAULT_SETUPS = tuple([setup for setup in MlSetup])
DEFAULT_DATASETS = ("adni", "ppmi", "qpn")
DEFAULT_N_SPLITS = 1
DEFAULT_N_ITER_NULL = 1

MODEL_MAP = {
    MlTarget.AGE: partial(Ridge),
    MlTarget.COG_DECLINE: partial(
        LogisticRegression, class_weight="balanced", warm_start=True, solver="lbfgs"
    ),
    MlTarget.DIAGNOSIS: partial(
        LogisticRegression, class_weight="balanced", warm_start=True, solver="lbfgs"
    ),
    MlTarget.MMSE: partial(Ridge),
}

warnings.filterwarnings(action="ignore", message="X has feature names")


class KnownError(Exception):
    pass


class SklearnWorkflow:
    def __init__(
        self,
        dpath_data: Path,
        dpath_results: Path,
        data_tags: str,
        n_rounds: int = DEFAULT_N_ROUNDS,
        setups: Iterable[MlSetup] = DEFAULT_SETUPS,
        test_datasets: Iterable[str] = DEFAULT_DATASETS,
        n_splits: int = DEFAULT_N_SPLITS,
        n_iter_null: int = DEFAULT_N_ITER_NULL,
        random_state: Optional[int] = None,
        sloppy: bool = False,
        overwrite: bool = False,
    ):
        self.dpath_data = Path(dpath_data)
        self.dpath_results = Path(dpath_results)
        self.data_tags = data_tags
        self.n_rounds = n_rounds
        self.setups = setups
        self.test_datasets = test_datasets
        self.n_splits = n_splits
        self.n_iter_null = n_iter_null
        self.random_state = random_state
        self.sloppy = sloppy
        self.overwrite = overwrite

        self.target = get_target_from_tag(self.data_tags)
        target_cols = [self.target.value.upper()]
        self.target_cols = target_cols

        problem = ML_TARGET_TO_PROBLEM_MAP[get_target_from_tag(self.data_tags)]
        self.problem = problem

        model = self.get_model()
        self.model = model

        self.settings = deepcopy(locals())
        self.settings.pop("self")
        for key in (
            "dpath_data",
            "dpath_results",
            "model",
        ):
            self.settings[key] = str(self.settings[key])

        dpath_run_results = (
            self.dpath_results
            / datetime.datetime.now().strftime("%Y_%m_%d")
            / f"{self.data_tags}-{self.random_state}"
        )
        self.dpath_run_results = dpath_run_results

    def get_train_data(
        self,
        i_split: int,
        null: bool,
        setup: MlSetup,
        dataset: Optional[str] = None,
        squeeze: bool = True,
    ):
        if dataset is not None or setup == MlSetup.SILO:
            dataset_str = dataset
        elif setup == MlSetup.MEGA:
            dataset_str = "_".join(["mega"] + list(sorted(self.test_datasets)))

        fpath = (
            self.dpath_data
            / f"{dataset_str}-{self.data_tags}"
            / f"{dataset_str}-{self.data_tags}-{i_split}train.tsv"
        )

        X, y = load_Xy(
            fpath,
            target_cols=self.target_cols,
            setup=setup,
            dataset=dataset,
            datasets=self.test_datasets,
            null=null,
        )
        if squeeze:
            y = y.squeeze(axis=1)

        return X, y

    def get_test_data(self, i_split: int, setup: MlSetup):
        Xy_test_all = {}
        n_features = None
        n_targets = None
        for col_dataset in self.test_datasets:
            fpath = (
                self.dpath_data
                / f"{col_dataset}-{self.data_tags}"
                / f"{col_dataset}-{self.data_tags}-{i_split}test.tsv"
            )
            X_test, y_test = load_Xy(
                fpath,
                target_cols=self.target_cols,
                setup=setup,
                dataset=col_dataset,
                datasets=self.test_datasets,
            )

            if n_features is None:
                n_features = X_test.shape[1]
            else:
                assert n_features == X_test.shape[1], "Inconsistent number of features"

            if n_targets is None:
                n_targets = y_test.shape[1]
            else:
                assert n_targets == y_test.shape[1], "Inconsistent number of targets"

            Xy_test_all[col_dataset] = (X_test, y_test)

        return Xy_test_all, n_features, n_targets

    def get_model(self):
        # return MODEL_MAP[self.target](random_state=self.random_state)
        return make_pipeline(
            StandardScaler(), MODEL_MAP[self.target](random_state=self.random_state)
        )

    def get_results(
        self,
        i_split: int,
        i_iter: int,
        null: bool,
        setup: MlSetup,
        test_datasets_subset: Iterable[str],
        dataset: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        Xy_test_all, n_features, n_targets = self.get_test_data(
            i_split=i_split, setup=setup
        )

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
                for dataset in self.test_datasets:
                    X_train, y_train = self.get_train_data(
                        i_split=i_split, null=null, setup=setup, dataset=dataset
                    )

                    model = clone(self.model)
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
            X_train, y_train = self.get_train_data(
                i_split=i_split, null=null, setup=setup, dataset=dataset
            )

            model = clone(self.model)
            model.fit(X_train, y_train)

        # evaluate
        metrics_map = get_metrics_map(self.problem)
        for test_dataset in test_datasets_subset:
            X_test, y_test = Xy_test_all[test_dataset]

            y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics_map.items():
                yield {
                    "setup": setup.value,
                    "problem": self.problem.value,
                    "target": self.target.value,
                    # for silo, the test dataset is also the train dataset
                    "test_dataset": test_dataset,
                    "is_null": null,
                    "metric": metric_name,
                    "i_split": i_split,
                    "i_iter": i_iter,
                    "score": metric_func(y_test, y_pred),
                }

    def get_results_all_setups(
        self, n_iter: int, null: bool = False
    ) -> Generator[dict, None, None]:
        for setup in self.setups:
            for i_split in range(self.n_splits):
                for i_iter in range(n_iter):
                    print(
                        f"Running:\tsetup={setup.value}\t{i_split=}\t{i_iter=}\t{null=}"
                    )
                    if setup == MlSetup.SILO:
                        for dataset in self.test_datasets:
                            yield from (
                                self.get_results(
                                    i_split=i_split,
                                    i_iter=i_iter,
                                    null=null,
                                    setup=setup,
                                    dataset=dataset,
                                    test_datasets_subset=[dataset],
                                )
                            )
                    else:
                        yield from (
                            self.get_results(
                                i_split=i_split,
                                i_iter=i_iter,
                                null=null,
                                setup=setup,
                                test_datasets_subset=self.test_datasets,
                            )
                        )

    def run(self):
        # results paths
        fname_base_metrics = f"metrics-{self.n_splits}_splits-{self.n_iter_null}_null"
        fname_base_settings = f"settings-{self.n_splits}_splits-{self.n_iter_null}_null"
        if self.sloppy:
            fname_base_metrics += "-sloppy"
            fname_base_settings += "-sloppy"
        fpath_metrics = self.dpath_run_results / f"{fname_base_metrics}.tsv"
        fpath_settings = self.dpath_run_results / f"{fname_base_settings}.json"

        # save settings
        self.dpath_run_results.mkdir(parents=True, exist_ok=True)
        if fpath_settings.exists() and not self.overwrite:
            raise KnownError(
                f"Settings file already exists: {fpath_settings}. "
                "Use --overwrite to overwrite."
            )
        fpath_settings.write_text(json.dumps(self.settings, indent=4))

        # normal models
        data_results = []
        for results in self.get_results_all_setups(n_iter=1, null=False):
            # save the results as they are being obtained
            data_results.append(results)
            df_results = pd.DataFrame(data_results)
            if len(df_results) > 0:
                df_results.to_csv(fpath_metrics, sep="\t", index=False)

        # null models
        for results in self.get_results_all_setups(n_iter=self.n_iter_null, null=True):
            # save the results as they are being obtained
            data_results.append(results)
            df_results = pd.DataFrame(data_results)
            if len(df_results) > 0:
                df_results.to_csv(fpath_metrics, sep="\t", index=False)

        print(f"Results saved to {fpath_metrics}")


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
    "test_datasets",
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
@click.option(
    "--sloppy", is_flag=True, help="Run Fed-BioMed as fast as possible (for testing)"
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing results files")
def run_fedbiomed(**params):
    workflow = SklearnWorkflow(**params)
    workflow.run()


if __name__ == "__main__":
    try:
        run_fedbiomed()
    except KnownError as exception:
        click.echo(click.style(f"ERROR: {exception}", fg="red", bold=True))
        sys.exit(1)
