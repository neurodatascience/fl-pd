#!/usr/bin/env python

import datetime
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Generator, Iterable, Optional

import click
import pandas as pd
from yaml import load, Loader

# from fedbiomed.common.optimizers.optimizer import Optimizer
# from fedbiomed.common.optimizers.declearn import YogiModule as FedYogi
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

from fl_pd.io import load_Xy
from fl_pd.metrics import get_metrics_map
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    MlProblem,
    MlSetup,
    MlFramework,
)
from fl_pd.ml_spec import ML_TARGET_TO_PROBLEM_MAP, get_target_from_tag

DEFAULT_SETUPS = tuple([setup for setup in MlSetup])
DEFAULT_DATASETS = ("adni", "ppmi", "qpn")
DEFAULT_N_SPLITS = 1
DEFAULT_N_ITER_NULL = 1
DEFAULT_SGDC_LOSS = "log_loss"
DEFAULT_SGDR_LOSS = "squared_error"


class KnownError(Exception):
    pass


class FedbiomedWorkflow:
    def __init__(
        self,
        dpath_data: Path,
        dpath_results: Path,
        dpath_researcher: Path,
        fpath_config: Path,
        data_tags: str,
        framework: MlFramework,
        setups: Iterable[MlSetup] = DEFAULT_SETUPS,
        sgdc_loss: str = DEFAULT_SGDC_LOSS,
        sgdr_loss: str = DEFAULT_SGDR_LOSS,
        test_datasets: Iterable[str] = DEFAULT_DATASETS,
        n_splits: int = DEFAULT_N_SPLITS,
        n_iter_null: int = DEFAULT_N_ITER_NULL,
        random_state: Optional[int] = None,
        sloppy: bool = False,
        overwrite: bool = False,
    ):
        self.dpath_data = Path(dpath_data)
        self.dpath_results = Path(dpath_results)
        self.dpath_researcher = Path(dpath_researcher)
        self.fpath_config = Path(fpath_config)
        self.data_tags = data_tags
        self.framework = framework
        self.setups = setups
        self.sgdc_loss = sgdc_loss
        self.sgdr_loss = sgdr_loss
        self.test_datasets = test_datasets
        self.n_splits = n_splits
        self.n_iter_null = n_iter_null
        self.random_state = random_state
        self.sloppy = sloppy
        self.overwrite = overwrite

        target_cols = [get_target_from_tag(self.data_tags).value.upper()]
        self.target_cols = target_cols

        problem = ML_TARGET_TO_PROBLEM_MAP[get_target_from_tag(self.data_tags)]
        self.problem = problem

        fedbiomed_tags = [self.data_tags]
        self.fedbiomed_tags = fedbiomed_tags

        if not self.fpath_config.exists():
            raise KnownError(f"Config file not found: {self.fpath_config}")
        fbm_configs = load(self.fpath_config.read_text(), Loader)
        self.fbm_configs = fbm_configs

        self.settings = deepcopy(locals())
        self.settings.pop("self")
        for key in ("dpath_data", "dpath_results", "dpath_researcher", "fpath_config"):
            self.settings[key] = str(self.settings[key])
        self.settings["model_args"] = self.get_model_args(None, None)

        framework_tags = f"{self.framework.value}-{self.problem.value}"
        if self.framework == MlFramework.SKLEARN:
            if self.problem == MlProblem.CLASSIFICATION:
                framework_tags += f"-{self.sgdc_loss}"
            elif self.problem == MlProblem.REGRESSION:
                framework_tags += f"-{self.sgdr_loss}"
        dpath_run_results = (
            self.dpath_results
            / datetime.datetime.now().strftime("%Y_%m_%d")
            / f"{framework_tags}-{self.data_tags}-{self.random_state}"
        )
        self.dpath_run_results = dpath_run_results

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

    def get_model_args(self, n_features, n_targets, null=False):
        model_args = {
            "eta0": 0.05,  # NativeSklearnOptimizer
            # "learning_rate": 0.01,  # NativeSklearnOptimizer
            "random_state": self.random_state,
            "n_features": n_features,
            "n_targets": n_targets,
            "target_cols": self.target_cols,
            "shuffle": True,
            "problem_type": self.problem.value,
            "null": null,
        }
        if self.problem == MlProblem.CLASSIFICATION:
            model_args["n_classes"] = 2
            loss = self.sgdc_loss
        elif self.problem == MlProblem.REGRESSION:
            loss = self.sgdr_loss
        model_args["loss"] = loss
        return model_args

    def get_training_plan(self):
        # fmt: off
        if self.framework == MlFramework.SKLEARN:
            if self.problem == MlProblem.CLASSIFICATION:
                from fl_pd.training_plans.sklearn import SklearnClassifierTrainingPlan
                return SklearnClassifierTrainingPlan
            elif self.problem == MlProblem.REGRESSION:
                from fl_pd.training_plans.sklearn import SklearnRegressorTrainingPlan
                return SklearnRegressorTrainingPlan
        # fmt: on
        raise KnownError(
            f"No training plan found for: {self.problem=}, {self.framework=}"
        )

    def get_results(
        self,
        i_split: int,
        i_iter: int,
        null: bool,
        setup: MlSetup,
        test_datasets_subset: Iterable[str],
        nodes: Iterable[str],
    ) -> Generator[dict, None, None]:
        Xy_test_all, n_features, n_targets = self.get_test_data(
            i_split=i_split, setup=setup
        )

        fbm_training_config = self.fbm_configs[
            setup.value if not self.sloppy else "fast"
        ]

        # Fed-BioMed experiment
        experiment = Experiment(
            nodes=nodes,
            tags=self.fedbiomed_tags + [f"{i_split}train"],
            training_plan_class=self.get_training_plan(),
            model_args=self.get_model_args(n_features, n_targets, null=null),
            round_limit=fbm_training_config["rounds"],
            training_args=fbm_training_config["training_args"],
            aggregator=FedAverage(),
            # agg_optimizer=Optimizer(
            #     **fbm_training_config["training_args"]["optimizer_args"],
            #     modules=[FedYogi()],
            # ),
            node_selection_strategy=None,
            # config_path=self.dpath_researcher,  # seems to be causing auth problems
        )
        try:
            experiment.run()
        except SystemExit:
            return []

        # get final model
        experiment.training_plan().set_model_params(
            experiment.aggregated_params()[fbm_training_config["rounds"] - 1]["params"]
        )
        model = experiment.training_plan().model()

        # evaluate
        metrics_map = get_metrics_map(self.problem)
        for test_dataset in test_datasets_subset:
            X_test, y_test = Xy_test_all[test_dataset]

            if self.framework == MlFramework.SKLEARN:
                y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics_map.items():
                yield {
                    "setup": setup.value,
                    "problem": self.problem.value,
                    "framework": self.framework.value,
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
                    if setup == MlSetup.SILO:
                        for dataset in self.test_datasets:
                            yield from (
                                self.get_results(
                                    i_split=i_split,
                                    i_iter=i_iter,
                                    null=null,
                                    setup=setup,
                                    test_datasets_subset=[dataset],
                                    nodes=[f"NODE_{dataset.upper()}"],
                                )
                            )
                    else:
                        if setup == MlSetup.FEDERATED:
                            nodes = [
                                f"NODE_{dataset.upper()}"
                                for dataset in self.test_datasets
                            ]
                        elif setup == MlSetup.MEGA:
                            nodes = ["NODE_MEGA"]
                        yield from (
                            self.get_results(
                                i_split=i_split,
                                i_iter=i_iter,
                                null=null,
                                setup=setup,
                                test_datasets_subset=self.test_datasets,
                                nodes=nodes,
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
@click.argument(
    "dpath_researcher",
    type=click.Path(path_type=Path, file_okay=False),
    envvar="DPATH_FEDBIOMED_RESEARCHER",
)
@click.argument(
    "fpath_config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_FEDBIOMED_CONFIG",
)
@click.option("--tag", "data_tags", required=True)
@click.option(
    "--setup",
    "setups",
    type=click.Choice(MlSetup, case_sensitive=False),
    multiple=True,
    default=DEFAULT_SETUPS,
)
@click.option(
    "--framework",
    type=click.Choice(MlFramework, case_sensitive=False),
    required=True,
)
@click.option(
    "--sgdc-loss",
    type=str,
    default=DEFAULT_SGDC_LOSS,
    help="Loss for Sklearn SGDClassifier",
)
@click.option(
    "--sgdr-loss",
    type=str,
    default=DEFAULT_SGDR_LOSS,
    help="Loss for Sklearn SGDRegressor",
)
@click.option(
    "--dataset",
    "test_datasets",
    type=str,
    multiple=True,
    default=DEFAULT_DATASETS,
    help="Dataset to use for testing (train datasets determined by active nodes and MlSetup).",
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
    workflow = FedbiomedWorkflow(**params)
    workflow.run()


if __name__ == "__main__":
    try:
        run_fedbiomed()
    except KnownError as exception:
        click.echo(click.style(f"ERROR: {exception}", fg="red", bold=True))
        sys.exit(1)
