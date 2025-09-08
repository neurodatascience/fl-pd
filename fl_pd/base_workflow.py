import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Iterable, Optional

import pandas as pd

from fl_pd.io import load_Xy, get_dpath_latest
from fl_pd.utils.constants import MlFramework, MlSetup
from fl_pd.metrics import get_metrics_map
from fl_pd.ml_spec import ML_TARGET_TO_PROBLEM_MAP, get_target_from_tag


DEFAULT_DATASETS = ("adni", "calgary", "pad", "ppmi", "qpn")
DEFAULT_N_SPLITS = 1
DEFAULT_N_ITER_NULL = 1
DEFAULT_SETUPS = tuple([setup for setup in MlSetup])


class KnownError(Exception):
    pass


class BaseWorkflow(ABC):
    def __init__(
        self,
        dpath_data: Path,
        dpath_results: Path,
        data_tags: str,
        framework: MlFramework,
        setups: Iterable[MlSetup] = DEFAULT_SETUPS,
        datasets: Iterable[str] = DEFAULT_DATASETS,
        n_splits: int = DEFAULT_N_SPLITS,
        n_iter_null: int = DEFAULT_N_ITER_NULL,
        random_state: Optional[int] = None,
        overwrite: bool = False,
    ):
        super().__init__()

        self.dpath_data = Path(dpath_data)
        self.dpath_results = Path(dpath_results)
        self.data_tags = data_tags
        self.framework = framework
        self.setups = setups
        self.datasets = datasets
        self.n_splits = n_splits
        self.n_iter_null = n_iter_null
        self.random_state = random_state
        self.overwrite = overwrite

        self.target = get_target_from_tag(self.data_tags)
        self.problem = ML_TARGET_TO_PROBLEM_MAP[self.target]
        self.target_col = self.target.value.upper()
        self.dpath_run_results = None  # to be set in run()

    @property
    def results_suffix(self) -> str:
        return f"{self.n_splits}_splits-{self.n_iter_null}_null-{self.random_state}"

    @property
    def settings(self) -> dict:
        settings = self.__dict__.copy()
        for key, value in settings.items():
            if isinstance(value, tuple):
                continue
            if isinstance(value, Path):
                value = value.resolve()
            settings[key] = str(value)
        return settings

    def get_test_data(self, i_split: int, setup: MlSetup):
        Xy_test_all = {}
        n_features = None
        n_targets = None
        for col_dataset in self.datasets:
            fpath = (
                self.dpath_data
                / f"{col_dataset}-{self.data_tags}"
                / f"{col_dataset}-{self.data_tags}-{i_split}test.tsv"
            )
            X_test, y_test = load_Xy(
                fpath,
                target_cols=[self.target_col],
                setup=setup,
                dataset=col_dataset,
                datasets=self.datasets,
            )

            if n_features is None:
                n_features = X_test.shape[1]
            else:
                assert (
                    n_features == X_test.shape[1]
                ), f"Inconsistent number of features ({n_features=} vs {X_test.shape[1]=}): {fpath}"

            if n_targets is None:
                n_targets = y_test.shape[1]
            else:
                assert n_targets == y_test.shape[1], "Inconsistent number of targets"

            Xy_test_all[col_dataset] = (X_test, y_test)

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

        return Xy_test_all, n_features, n_targets, sample_weight

    @abstractmethod
    def train(
        self,
        setup: MlSetup,
        n_features: int,
        n_targets: int,
        i_split: int,
        null: bool,
        train_dataset: str,
    ):
        pass

    def get_results(
        self,
        i_split: int,
        i_iter: int,
        null: bool,
        setup: MlSetup,
        train_dataset: str,
    ) -> Generator[dict, None, None]:
        Xy_test_all, n_features, n_targets, sample_weight = self.get_test_data(
            i_split=i_split, setup=setup
        )

        model = self.train(
            setup=setup,
            n_features=n_features,
            n_targets=n_targets,
            i_split=i_split,
            null=null,
            train_dataset=train_dataset,
        )

        metrics_map = get_metrics_map(self.problem)
        for test_dataset in Xy_test_all.keys():
            X_test, y_test = Xy_test_all[test_dataset]

            y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics_map.items():
                yield {
                    "setup": setup.value,
                    "problem": self.problem.value,
                    "target": self.target.value,
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
        for setup in self.setups:
            for i_split in range(self.n_splits):
                for i_iter in range(n_iter):
                    print(
                        f"Running:\tsetup={setup.value}\t{i_split=}\t{i_iter=}\t{null=}"
                    )
                    if setup == MlSetup.SILO:
                        for dataset in self.datasets:
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
                                train_dataset="-".join(sorted(self.datasets)),
                            )
                        )

    def run(self):
        # results paths
        self.dpath_run_results = (
            get_dpath_latest(self.dpath_results, use_today=True) / self.data_tags
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
