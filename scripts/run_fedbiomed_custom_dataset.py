#!/usr/bin/env python

import json
import sys
import pickle
from pathlib import Path
from typing import Generator, Iterable, Optional, Type

import click
import pandas as pd
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.dataset import CustomDataset

from fl_pd.custom_dataset import training_plan_factory
from fl_pd.io import get_dpath_latest
from fl_pd.metrics import get_metrics_map
from fl_pd.io import working_directory
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS, MlSetup, MlProblem

DEFAULT_N_SPLITS = 10
DEFAULT_N_ITER_NULL = 1
DEFAULT_SETUPS = tuple([setup for setup in MlSetup])
DEFAULT_STANDARDIZE = True

DEFAULT_TAG_FEDERATED = "federated"
DEFAULT_TAG_MEGA = "mega"
DEFAULT_TRAIN_DATASETS = ("adni", "calgary", "pad", "ppmi", "qpn")  # fedbiomed tags
DEFAULT_TEST_DATASETS = (
    "adni",
    "calgary",
    "pad",
    "ppmi",
    "qpn",
    # "tums",
)  # local datasets

DEFAULT_N_ROUNDS = 20
DEFAULT_N_UPDATES = 50
DEFAULT_BATCH_SIZE = 50
DEFAULT_SAVE_MODEL = False

# TODO null models for Federated have problems for age due to data distribution differences

TARGET_PROBLEM_MAP = {
    "nb:Age": MlProblem.REGRESSION,
    "nb:Diagnosis": MlProblem.CLASSIFICATION,
    "fl:cognitive_decline_status": MlProblem.CLASSIFICATION,
}


class KnownError(Exception):
    pass


class FedbiomedWorkflow:
    def __init__(
        self,
        target: str,
        dpath_data: Path,
        dpath_results: Path,
        dpath_fedbiomed: Path,
        dpath_stats: Path,
        train_datasets: Iterable[str] = DEFAULT_TRAIN_DATASETS,
        test_datasets: Iterable[str] = DEFAULT_TEST_DATASETS,
        setups: Iterable[MlSetup] = DEFAULT_SETUPS,
        standardize: bool = DEFAULT_STANDARDIZE,
        n_splits: int = DEFAULT_N_SPLITS,
        n_iter_null: int = DEFAULT_N_ITER_NULL,
        random_state: Optional[int] = None,
        save_model: bool = False,
        overwrite: bool = False,
        n_rounds: int = DEFAULT_N_ROUNDS,
        n_updates: int = DEFAULT_N_UPDATES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sloppy: bool = False,
        tag_federated: str = DEFAULT_TAG_FEDERATED,
        tag_mega: str = DEFAULT_TAG_MEGA,
    ):
        super().__init__()

        if sloppy:
            n_rounds = 1
            n_updates = 1
            batch_size = int(1e6)

        self.target = target
        self.dpath_data = Path(dpath_data)
        self.dpath_results = Path(dpath_results)
        self.dpath_stats = Path(dpath_stats)
        self.train_datasets: list = list(train_datasets)
        self.test_datasets = test_datasets
        self.setups = setups
        self.standardize = standardize
        self.n_splits = n_splits
        self.n_iter_null = n_iter_null
        self.random_state = random_state
        self.save_model = save_model
        self.overwrite = overwrite
        self.tag_federated = tag_federated
        self.tag_mega = tag_mega
        self.dpath_fedbiomed = Path(dpath_fedbiomed)
        self.n_rounds = n_rounds
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.sloppy = sloppy

        self.dpath_run_results = None  # to be set in run()
        self.problem = TARGET_PROBLEM_MAP[self.target]

    @property
    def results_suffix(self) -> str:
        suffix_components = []
        if self.standardize:
            suffix_components.append("standardize")
        else:
            suffix_components.append("no_standardize")
        suffix_components.extend(
            [
                f"{self.n_splits}_splits",
                f"{self.n_iter_null}_null",
                f"{self.random_state}",
            ]
        )
        suffix = "-".join(suffix_components)

        if self.sloppy:
            suffix += "-sloppy"

        framework_tags = "fbm"

        tags = [
            framework_tags,
            f"{self.n_rounds}_rounds",
            f"{self.n_updates}_updates",
            f"{self.batch_size}_batch_size",
            suffix,
        ]

        return "-".join(tags)

    @property
    def settings(self) -> dict:
        settings = self.__dict__.copy()
        for key, value in settings.items():
            if isinstance(value, (tuple, bool, int, float)) or (
                isinstance(value, list) and all(isinstance(v, str) for v in value)
            ):
                continue
            if isinstance(value, Path):
                value = value.resolve()
            settings[key] = str(value)
        settings["model_args"] = self._get_model_args(None, None)
        return settings

    @property
    def training_plan(self) -> Type[BaseTrainingPlan]:
        return training_plan_factory(self.target)

    def get_fname_stats(self, setup: MlSetup, i_split: int, train_dataset: str) -> Path:
        if setup == MlSetup.SILO:
            dataset_str = train_dataset
        else:
            # TODO maybe different for federated if not all datasets are known
            dataset_str = "_".join([self.tag_mega] + sorted(self.train_datasets))

        fname_stats = f"{dataset_str}-{self.target}-{self.n_splits}splits-rng{self.random_state}-{i_split}.tsv"
        return fname_stats

    def get_test_data(self, i_split: int, setup: MlSetup, train_dataset: str):
        Xy_test_all = {}

        for dataset_name in self.test_datasets:

            dpath_dataset: Path = (
                get_dpath_latest(self.dpath_data) / dataset_name
            )  # Nipoppy root
            if not dpath_dataset.exists():
                raise KnownError(f"Local dataset not found: {dpath_dataset}")

            if self.standardize:
                fname_stats = self.get_fname_stats(setup, i_split, train_dataset)
            else:
                fname_stats = None

            dataset: CustomDataset = self.training_plan.dataset_factory(
                target=self.target,
                i_split=i_split,
                n_splits=self.n_splits,
                train=False,
                random_state=self.random_state,
                fname_stats=fname_stats,
            )
            dataset.path = dpath_dataset
            X_test, y_test = dataset.read()

            Xy_test_all[dataset_name] = (X_test, y_test)

        # for combined case
        sample_weight = pd.concat(
            [
                pd.Series([1 / len(Xy_test_all) / X.shape[0]] * X.shape[0])
                for X, _ in Xy_test_all.values()
            ],
            axis="index",
            ignore_index=True,
        )

        all_datasets_str = "-".join(sorted(Xy_test_all.keys()))
        Xy_test_all[all_datasets_str] = (
            pd.concat([X for X, _ in Xy_test_all.values()], axis="index"),
            pd.concat([y for _, y in Xy_test_all.values()], axis="index"),
        )

        Xy_test_all[f"{all_datasets_str}-weighted"] = Xy_test_all[all_datasets_str]

        n_features = Xy_test_all[all_datasets_str][0].shape[1]
        return Xy_test_all, n_features, sample_weight

    def _get_model_args(
        self, n_features: int, i_split: int, null=False, fname_stats=None
    ):
        model_args = {
            # fedbiomed
            "eta0": 0.05,
            "random_state": self.random_state,
            "n_features": n_features,
            "n_classes": 2,  # ignored in regression tasks?
            # model
            "learning_rate": "invscaling",
            "penalty": "l2",
            # data-loading
            "target": self.target,
            "i_split": i_split,
            "n_splits": self.n_splits,
            "null": null,
            "fname_stats": str(fname_stats) if fname_stats is not None else None,
            "shuffle": True,
        }
        return model_args

    def train(
        self,
        setup: MlSetup,
        n_features: int,
        i_split: int,
        null: bool,
        train_dataset: str,
    ):
        tags = [self.target]
        if setup == MlSetup.SILO:
            tags.append(train_dataset)
        elif setup == MlSetup.FEDERATED:
            tags.append(self.tag_federated)
        elif setup == MlSetup.MEGA:
            tags.append("_".join([self.tag_mega] + sorted(self.train_datasets)))

        if self.standardize:
            fname_stats = self.get_fname_stats(setup, i_split, train_dataset)
        else:
            fname_stats = None

        # Fed-BioMed experiment
        with working_directory(self.dpath_fedbiomed):
            from fedbiomed.researcher.federated_workflows import Experiment
            from fedbiomed.researcher.aggregators.fedavg import FedAverage

            experiment = Experiment(
                tags=tags,
                training_plan_class=self.training_plan,
                model_args=self._get_model_args(
                    n_features=n_features,
                    i_split=i_split,
                    null=null,
                    fname_stats=fname_stats,
                ),
                round_limit=self.n_rounds,
                training_args={
                    "num_updates": self.n_updates,
                    "loader_args": {"batch_size": self.batch_size},
                },
                aggregator=FedAverage(),
                node_selection_strategy=None,
            )
            try:
                experiment.run()
            except SystemExit:
                return None

        # get final model
        experiment.training_plan().set_model_params(
            experiment.aggregated_params()[self.n_rounds - 1]["params"]
        )
        return experiment.training_plan().model()

    def get_results(
        self,
        i_split: int,
        i_iter: int,
        null: bool,
        setup: MlSetup,
        train_dataset: str,
    ) -> Generator[dict, None, None]:
        Xy_test_all, n_features, sample_weight = self.get_test_data(
            i_split=i_split, setup=setup, train_dataset=train_dataset
        )

        model = self.train(
            setup=setup,
            n_features=n_features,
            i_split=i_split,
            null=null,
            train_dataset=train_dataset,
        )

        if model is None:
            return []  # skip evaluation if model training failed

        if self.save_model:
            prefix_model = self.results_suffix
            if null:
                prefix_model = f"{prefix_model}-null"
            prefix_model = f"{prefix_model}-{i_split}-{i_iter}"
            if setup == MlSetup.SILO:
                prefix_model += f"-{setup.value}_{train_dataset}"
            else:
                prefix_model += f"-{setup.value}"
            fpath_model = self.dpath_run_results / f"{prefix_model}.pkl"
            with open(fpath_model, "wb") as file_model:
                pickle.dump(model, file_model)
            print(f"Model saved to {fpath_model}")

        metrics_map = get_metrics_map(self.problem)
        for test_dataset in Xy_test_all.keys():
            X_test, y_test = Xy_test_all[test_dataset]

            y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics_map.items():
                yield {
                    "setup": setup.value,
                    "problem": self.problem.value,
                    "target": self.target,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "is_null": null,
                    "metric": metric_name,
                    "i_split": i_split,
                    "i_iter": i_iter,
                    "score": metric_func(
                        y_test,
                        y_pred,
                        sample_weight=(
                            sample_weight if "weighted" in test_dataset else None
                        ),
                    ),
                }

    def get_results_all_setups(
        self, n_iter: int, null: bool = False
    ) -> Generator[dict, None, None]:
        for i_split in range(self.n_splits):
            for setup in self.setups:
                for i_iter in range(n_iter):
                    print(
                        f"Running:\tsetup={setup.value}\t{i_split=}\t{i_iter=}\t{null=}"
                    )
                    if setup == MlSetup.SILO:
                        for dataset in self.train_datasets:
                            yield from (
                                self.get_results(
                                    i_split=i_split,
                                    i_iter=i_iter,
                                    null=null,
                                    setup=setup,
                                    train_dataset=dataset,
                                )
                            )
                    else:
                        yield from (
                            self.get_results(
                                i_split=i_split,
                                i_iter=i_iter,
                                null=null,
                                setup=setup,
                                train_dataset="-".join(sorted(self.train_datasets)),
                            )
                        )

    def run(self):
        # results paths
        self.dpath_run_results = (
            get_dpath_latest(self.dpath_results, use_today=True) / self.target
        )
        fpath_metrics = self.dpath_run_results / f"{self.results_suffix}.tsv"
        fpath_settings = self.dpath_run_results / f"{self.results_suffix}.json"

        # save settings
        self.dpath_run_results.mkdir(parents=True, exist_ok=True)
        if fpath_metrics.exists() and not self.overwrite:
            raise KnownError(
                f"Metrics file already exists: {fpath_metrics}. "
                "Use --overwrite to overwrite."
            )
        self.settings["dpath_run_results"] = str(self.dpath_run_results)
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
@click.argument(
    "dpath_fedbiomed",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="DPATH_FEDBIOMED",
)
@click.argument(
    "dpath_stats",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_STATS",
)
@click.option("--target", type=click.Choice(TARGET_PROBLEM_MAP.keys()))
@click.option(
    "--train-dataset", "train_datasets", multiple=True, default=DEFAULT_TRAIN_DATASETS
)
@click.option(
    "--test-dataset", "test_datasets", multiple=True, default=DEFAULT_TEST_DATASETS
)
@click.option(
    "--setup",
    "setups",
    type=click.Choice(MlSetup, case_sensitive=False),
    multiple=True,
    default=DEFAULT_SETUPS,
)
@click.option("--tag-federated", type=str, default=DEFAULT_TAG_FEDERATED)
@click.option("--tag-mega", type=str, default=DEFAULT_TAG_MEGA)
@click.option("--n-rounds", type=click.IntRange(min=1), default=DEFAULT_N_ROUNDS)
@click.option("--n-updates", type=click.IntRange(min=1), default=DEFAULT_N_UPDATES)
@click.option("--batch-size", type=click.IntRange(min=1), default=DEFAULT_BATCH_SIZE)
@click.option(
    "--standardize/--no-standardize",
    default=DEFAULT_STANDARDIZE,
    help="Standardize train and test data based on train data mean/std",
)
@click.option("--n-splits", type=click.IntRange(min=1), default=DEFAULT_N_SPLITS)
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
    "--save-model/--no-save-model",
    default=DEFAULT_SAVE_MODEL,
    help="Whether to save the trained model(s) to a file",
)
@click.option(
    "--sloppy", is_flag=True, help="Run Fed-BioMed as fast as possible (for testing)"
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing results files")
def run_fedbiomed(**params):
    workflow = FedbiomedWorkflow(**params)
    workflow.run()


if __name__ == "__main__":
    try:
        run_fedbiomed()
    except KnownError as exception:
        click.echo(click.style(f"ERROR: {exception}", fg="red", bold=True))
        sys.exit(1)
