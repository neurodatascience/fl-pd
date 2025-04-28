#!/usr/bin/env python

import datetime
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import click
import pandas as pd
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

from fl_pd.io import load_Xy
from fl_pd.metrics import get_metrics_map
from fl_pd.utils.cli import CLICK_CONTEXT_SETTINGS
from fl_pd.utils.enums import MlProblem, MlSetup, MlFramework

DEFAULT_DATASETS = ("adni", "ppmi", "qpn")

# TODO move to config file?
FEDBIOMED_ROUNDS = 1
TRAINING_ARGS = {
    "epochs": 100,
    "loader_args": {
        "batch_size": 128,
    },
    # "epochs": 1,
    # "loader_args": {
    #     "batch_size": 10000,
    # },
}


class FedbiomedWorkflow:
    def __init__(
        self,
        dpath_data: Path,
        dpath_results: Path,
        dpath_researcher: Path,
        setups: Iterable[MlSetup],
        problem: MlProblem,
        framework: MlFramework,
        min_age: int,
        test_datasets: Iterable[str],
        n_splits: int,
        random_state: int,
    ):
        self.dpath_data = dpath_data
        self.dpath_results = dpath_results
        self.dpath_researcher = dpath_researcher
        self.setups = setups
        self.problem = problem
        self.framework = framework
        self.min_age = min_age
        self.test_datasets = test_datasets
        self.n_splits = n_splits
        self.random_state = random_state

        fedbiomed_tags = []
        if self.problem == MlProblem.CLASSIFICATION:
            target_cols = ["COG_DECLINE"]
            data_tags = "decline-age-case-aparc"
            fedbiomed_tags.append("cog_decline")
        elif self.problem == MlProblem.REGRESSION:
            target_cols = ["AGE"]
            data_tags = "age-sex-hc-aseg"
            fedbiomed_tags.append("brain_age")
        self.target_cols = target_cols
        self.data_tags = data_tags
        self.fedbiomed_tags = fedbiomed_tags

        self.config = deepcopy(locals())

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

    def get_model_args(self, n_features, n_targets):
        model_args = {
            "random_state": self.random_state,
            "n_features": n_features,
            "n_targets": n_targets,
            "target_cols": self.target_cols,
            "min_age": self.min_age,
            "shuffle": True,
            "problem_type": self.problem.value,
        }
        if self.problem == MlProblem.CLASSIFICATION:
            model_args["n_classes"] = 2
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
        raise ValueError(
            f"No training plan found for: {self.problem=}, {self.framework=}"
        )

    def get_results(
        self,
        i_split,
        setup: MlSetup,
        test_datasets_subset: Iterable[str],
        nodes: Iterable[str],
    ):
        Xy_test_all, n_features, n_targets = self.get_test_data(
            i_split=i_split, setup=setup
        )

        # Fed-BioMed experiment
        experiment = Experiment(
            nodes=nodes,
            tags=self.fedbiomed_tags + [f"{i_split}train"],
            training_plan_class=self.get_training_plan(),
            model_args=self.get_model_args(n_features, n_targets),
            training_args=TRAINING_ARGS,
            round_limit=FEDBIOMED_ROUNDS,
            aggregator=FedAverage(),
            node_selection_strategy=None,
            config_path=self.dpath_researcher,
        )
        try:
            experiment.run()
        except SystemExit:
            return []

        # get final model
        experiment.training_plan().set_model_params(
            experiment.aggregated_params()[FEDBIOMED_ROUNDS - 1]["params"]
        )
        model = experiment.training_plan().model()

        # evaluate
        metrics_map = get_metrics_map(self.problem)
        data_results = []
        for test_dataset in test_datasets_subset:
            X_test, y_test = Xy_test_all[test_dataset]

            if self.framework == MlFramework.SKLEARN:
                y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics_map.items():
                data_results.append(
                    {
                        "setup": setup.value,
                        "problem": self.problem.value,
                        "framework": self.framework.value,
                        # for silo, the test dataset is also the train dataset
                        "test_dataset": test_dataset,
                        "metric": metric_name,
                        "i_split": i_split,
                        "score": metric_func(y_test, y_pred),
                    }
                )
        return data_results

    def run(self):
        data_results = []
        for setup in self.setups:
            for i_split in range(self.n_splits):

                if setup == MlSetup.SILO:
                    for dataset in self.test_datasets:
                        data_results.extend(
                            self.get_results(
                                i_split=i_split,
                                setup=setup,
                                test_datasets_subset=[dataset],
                                nodes=[f"NODE_{dataset.upper()}"],
                            )
                        )
                else:
                    if setup == MlSetup.FEDERATED:
                        nodes = [
                            f"NODE_{dataset.upper()}" for dataset in self.test_datasets
                        ]
                    elif setup == MlSetup.MEGA:
                        nodes = ["NODE_MEGA"]
                    data_results.extend(
                        self.get_results(
                            i_split=i_split,
                            setup=setup,
                            test_datasets_subset=self.test_datasets,
                            nodes=nodes,
                        )
                    )

        df_results = pd.DataFrame(data_results)
        fpath_out = (
            self.dpath_results
            / self.framework.value
            / datetime.datetime.now().strftime("%Y_%m_%d")
            / f"results-{self.problem.value}-{self.data_tags}-{self.min_age}-{self.n_splits}-{self.random_state}.tsv"
        )
        fpath_out.parent.mkdir(parents=True, exist_ok=True)
        if len(df_results) > 0:
            df_results.to_csv(fpath_out, sep="\t", index=False)
            print(f"Results saved to {fpath_out}")

        # TODO save config (model args, training args, number of rounds, etc.)


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
@click.option(
    "--setup",
    "setups",
    type=click.Choice(MlSetup, case_sensitive=False),
    multiple=True,
    default=[setup for setup in MlSetup],
)
@click.option(
    "--problem",
    type=click.Choice(MlProblem, case_sensitive=False),
    required=True,
)
@click.option(
    "--framework",
    type=click.Choice(MlFramework, case_sensitive=False),
    required=True,
)
@click.option("--min-age", type=int, default=55)
@click.option(
    "--dataset",
    "test_datasets",
    type=str,
    multiple=True,
    default=DEFAULT_DATASETS,
    help="Dataset to use for testing (train datasets determined by active nodes).",
)
@click.option("--n-splits", type=click.IntRange(min=1), default=1)
@click.option(
    "--random-state",
    type=int,
    envvar="RANDOM_SEED",
    help="Random state for reproducibility",
)
def run_fedbiomed(**params):
    workflow = FedbiomedWorkflow(**params)
    workflow.run()


if __name__ == "__main__":
    run_fedbiomed()
