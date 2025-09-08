#!/usr/bin/env python

import sys
from functools import cached_property
from pathlib import Path

import click
from yaml import load, Loader

from fl_pd.base_workflow import (
    BaseWorkflow,
    KnownError,
    DEFAULT_DATASETS,
    DEFAULT_N_ITER_NULL,
    DEFAULT_N_SPLITS,
    DEFAULT_SETUPS,
)
from fl_pd.io import working_directory
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    MlProblem,
    MlSetup,
    MlFramework,
)

DEFAULT_SGDC_LOSS = "log_loss"
DEFAULT_SGDR_LOSS = "squared_error"


class FedbiomedWorkflow(BaseWorkflow):
    def __init__(
        self,
        dpath_fedbiomed: Path,
        fpath_config: Path,
        sgdc_loss: str = DEFAULT_SGDC_LOSS,
        sgdr_loss: str = DEFAULT_SGDR_LOSS,
        sloppy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dpath_fedbiomed = Path(dpath_fedbiomed)
        self.fpath_config = Path(fpath_config)
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
            if self.problem == MlProblem.CLASSIFICATION:
                framework_tags += f"-{self.sgdc_loss}"
            elif self.problem == MlProblem.REGRESSION:
                framework_tags += f"-{self.sgdr_loss}"

        return f"{framework_tags}-{suffix}"

    @property
    def settings(self) -> dict:
        settings = super().settings
        settings["model_args"] = self._get_model_args(None, None)
        return settings

    @cached_property
    def _fbm_configs(self) -> dict:
        if not self.fpath_config.exists():
            raise KnownError(f"Config file not found: {self.fpath_config}")
        return load(self.fpath_config.read_text(), Loader)

    def _get_model_args(self, n_features, n_targets, null=False):
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
        fbm_training_config = self._fbm_configs[
            setup.value if not self.sloppy else "fast"
        ]

        if setup == MlSetup.SILO:
            nodes = [f"NODE_{train_dataset.upper()}"]
        elif setup == MlSetup.FEDERATED:
            nodes = [f"NODE_{dataset.upper()}" for dataset in self.datasets]
        elif setup == MlSetup.MEGA:
            nodes = ["NODE_MEGA"]

        # Fed-BioMed experiment
        with working_directory(self.dpath_fedbiomed):
            # from fedbiomed.common.optimizers.optimizer import Optimizer
            # from fedbiomed.common.optimizers.declearn import YogiModule as FedYogi
            from fedbiomed.researcher.federated_workflows import Experiment
            from fedbiomed.researcher.aggregators.fedavg import FedAverage

            experiment = Experiment(
                nodes=nodes,
                tags=self.fedbiomed_tags + [f"{i_split}train"],
                training_plan_class=self._get_training_plan(),
                model_args=self._get_model_args(n_features, n_targets, null=null),
                round_limit=fbm_training_config["rounds"],
                training_args=fbm_training_config["training_args"],
                aggregator=FedAverage(),
                # agg_optimizer=Optimizer(
                #     **fbm_training_config["training_args"]["optimizer_args"],
                #     modules=[FedYogi()],
                # ),
                node_selection_strategy=None,
            )
            try:
                experiment.run()
            except SystemExit:
                return []

        # get final model
        experiment.training_plan().set_model_params(
            experiment.aggregated_params()[fbm_training_config["rounds"] - 1]["params"]
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
