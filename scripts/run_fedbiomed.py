#!/usr/bin/env python

import sys
from pathlib import Path

import click

from fl_pd.base_workflow import (
    BaseWorkflow,
    KnownError,
    DEFAULT_DATASETS,
    DEFAULT_N_ITER_NULL,
    DEFAULT_N_SPLITS,
    DEFAULT_SETUPS,
    DEFAULT_STANDARDIZE,
)
from fl_pd.io import working_directory
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    MlProblem,
    MlSetup,
    MlFramework,
)

DEFAULT_N_ROUNDS = 20
DEFAULT_N_UPDATES = 20
DEFAULT_BATCH_SIZE = 50
DEFAULT_SGDC_LOSS = "log_loss"
DEFAULT_SGDR_LOSS = "squared_error"
DEFAULT_SAVE_MODEL = False


class FedbiomedWorkflow(BaseWorkflow):
    def __init__(
        self,
        dpath_fedbiomed: Path,
        n_rounds: int = DEFAULT_N_ROUNDS,
        n_updates: int = DEFAULT_N_UPDATES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sgdc_loss: str = DEFAULT_SGDC_LOSS,
        sgdr_loss: str = DEFAULT_SGDR_LOSS,
        sloppy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if sloppy:
            n_rounds = 1
            n_updates = 1
            batch_size = int(1e6)

        self.dpath_fedbiomed = Path(dpath_fedbiomed)
        self.n_rounds = n_rounds
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.sgdc_loss = sgdc_loss
        self.sgdr_loss = sgdr_loss
        self.sloppy = sloppy

        self.fedbiomed_tags = [self.data_tags]

    @property
    def results_suffix(self) -> str:
        suffix = super().results_suffix
        if self.sloppy:
            suffix += "-sloppy"

        framework_tags = "fbm"
        if self.framework == MlFramework.SKLEARN:
            framework_tags += "_sklearn"

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
        settings = super().settings
        settings["model_args"] = self._get_model_args(None, None)
        return settings

    def _get_model_args(self, n_features, n_targets, null=False, fpath_stats=None):
        model_args = {
            "eta0": 0.05,  # NativeSklearnOptimizer
            "learning_rate": "invscaling",  # NativeSklearnOptimizer
            "random_state": self.random_state,
            "n_features": n_features,
            "n_targets": n_targets,
            "target_cols": [self.target_col],
            "shuffle": True,
            "problem_type": self.problem.value,
            "null": null,
            "fpath_stats": str(fpath_stats) if fpath_stats is not None else None,
            "penalty": "l2",
        }
        if self.problem == MlProblem.CLASSIFICATION:
            model_args["n_classes"] = 2
            loss = self.sgdc_loss
        elif self.problem == MlProblem.REGRESSION:
            loss = self.sgdr_loss
        model_args["loss"] = loss
        return model_args

    def _get_training_plan(self):
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

    def train(
        self,
        setup: MlSetup,
        n_features: int,
        n_targets: int,
        i_split: int,
        null: bool,
        train_dataset: str,
    ):
        if setup == MlSetup.SILO:
            nodes = [f"NODE_{train_dataset.upper()}"]
        elif setup == MlSetup.FEDERATED:
            nodes = [f"NODE_{dataset.upper()}" for dataset in self.datasets]
        elif setup == MlSetup.MEGA:
            nodes = ["NODE_MEGA"]

        if self.standardize:
            fpath_stats = self.get_fpath_stats(setup, i_split, train_dataset)
        else:
            fpath_stats = None

        # Fed-BioMed experiment
        with working_directory(self.dpath_fedbiomed):
            from fedbiomed.researcher.federated_workflows import Experiment
            from fedbiomed.researcher.aggregators.fedavg import FedAverage

            experiment = Experiment(
                nodes=nodes,
                tags=self.fedbiomed_tags + [f"{i_split}train"],
                training_plan_class=self._get_training_plan(),
                model_args=self._get_model_args(
                    n_features, n_targets, null=null, fpath_stats=fpath_stats
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
@click.option("--n-rounds", type=click.IntRange(min=1), default=DEFAULT_N_ROUNDS)
@click.option("--n-updates", type=click.IntRange(min=1), default=DEFAULT_N_UPDATES)
@click.option("--batch-size", type=click.IntRange(min=1), default=DEFAULT_BATCH_SIZE)
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
    "--standardize/--no-standardize",
    default=DEFAULT_STANDARDIZE,
    help="Standardize train and test data based on train data mean/std",
)
@click.option(
    "--dataset",
    "datasets",
    type=str,
    multiple=True,
    default=DEFAULT_DATASETS,
    help="Dataset to use for testing (train datasets determined by active nodes and MlSetup).",
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
