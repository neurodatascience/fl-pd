#!/usr/bin/env python

import json
from pathlib import Path
from typing import Tuple
import warnings

import click
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_pd.custom_dataset import training_plan_factory


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--target", type=str, required=True)
@click.option(
    "--dataset",
    "dataset_tags_and_paths",
    type=click.Tuple([str, Path]),
    nargs=2,
    multiple=True,
    required=False,
)
@click.option(
    "--output", "dpath_out", type=Path, required=True, envvar="DPATH_FL_STATS"
)
@click.option(
    "--mega",
    "dpath_mega",
    type=Path,
    required=False,
    envvar="DPATH_FL_DATA_LATEST_MEGA",
)
@click.option("--federated", type=str, multiple=True)
@click.option("--n-splits", type=int, default=10)
@click.option("--random-state", type=int, envvar="RANDOM_SEED")
def get_statistics(
    target,
    dataset_tags_and_paths: Tuple[str, Path],
    dpath_out: Path,
    dpath_mega: Path,
    federated: Tuple[str],
    n_splits,
    random_state,
):
    if not dataset_tags_and_paths and not dpath_mega and not federated:
        raise ValueError(
            "Must specify at least one of --dataset, --mega, or --federated."
        )
    dataset_tags_and_paths = list(dataset_tags_and_paths)
    if dpath_mega is not None:
        tag_mega = "_".join(
            ["mega"] + sorted([tag for tag, _ in dataset_tags_and_paths])
        )
        path_mega = dpath_mega / f"{tag_mega}-{target}.tsv"
        if path_mega.exists():
            mega_tag_and_path = (tag_mega, path_mega)
            if mega_tag_and_path not in dataset_tags_and_paths:
                dataset_tags_and_paths.append(mega_tag_and_path)

    for i_split in range(n_splits):
        suffix = f"-{target}-{n_splits}splits-rng{random_state}-{i_split}.tsv"

        for dataset_tag, dataset_path in dataset_tags_and_paths:
            dataset = training_plan_factory(target).dataset_factory(
                target=target,
                i_split=i_split,
                n_splits=n_splits,
                random_state=random_state,
                train=True,
                null=False,
                fname_stats=None,
            )

            dataset.path = str(dataset_path)
            X, _ = dataset.read()

            df_stats: pd.DataFrame = X.describe()

            dpath_out.mkdir(parents=True, exist_ok=True)
            fpath_stats = dpath_out / f"{dataset_tag}{suffix}"
            df_stats.to_csv(fpath_stats, sep="\t")
            print(f"Saved stats to {fpath_stats}")

            if dataset_path.is_file():
                fpath_global_config = dataset_path.parent / "global_config.json"
            else:
                fpath_global_config = dataset_path / "global_config.json"
            if not fpath_global_config.exists():
                warnings.warn(
                    f"Global config file not found at {fpath_global_config}, creating one."
                )
                fpath_global_config.write_text(
                    json.dumps({"CUSTOM": {"FL_PD": {}}}, indent=4)
                )
            global_config = json.loads(fpath_global_config.read_text())
            try:
                dpath_stats_old = global_config["CUSTOM"]["FL_PD"]["STATS"]
            except KeyError:
                dpath_stats_old = None

            if dpath_stats_old != dpath_out:
                global_config["CUSTOM"]["FL_PD"]["STATS"] = str(dpath_out)
                fpath_global_config.write_text(json.dumps(global_config, indent=4))
                print(f"Updated global config file at {fpath_global_config}")

        if federated:
            dfs_stats = []
            for dataset_tag in federated:
                fpath_stats = dpath_out / f"{dataset_tag}{suffix}"
                if not fpath_stats.exists():
                    raise FileNotFoundError(
                        f"Stats file for federated dataset not found at {fpath_stats}"
                    )
                df_stats = pd.read_csv(fpath_stats, sep="\t", index_col=0)
                dfs_stats.append(df_stats)

            df_stats_combined = pd.concat(dfs_stats, axis="index")
            df_stats_federated = pd.DataFrame(
                columns=df_stats_combined.columns,
                index=df_stats_combined.index.unique(),
            )
            df_stats_federated.loc["count", :] = (
                df_stats_combined.loc["count", :].sum().astype(int)
            )
            df_stats_federated.loc["mean", :] = (
                df_stats_combined.loc["mean", :].reset_index(drop=True)
                * df_stats_combined.loc["count", :].reset_index(drop=True)
            ).sum() / df_stats_federated.loc["count", :]
            df_stats_federated.loc["std", :] = (
                (
                    (df_stats_combined.loc["std", :].reset_index(drop=True) ** 2)
                    * (df_stats_combined.loc["count", :].reset_index(drop=True) - 1)
                ).sum()
                / (df_stats_federated.loc["count", :] - 1)
            ) ** 0.5

            dpath_out.mkdir(parents=True, exist_ok=True)
            fpath_stats = dpath_out / f"federated_{'_'.join(sorted(federated))}{suffix}"
            df_stats_federated.to_csv(fpath_stats, sep="\t")
            print(f"Saved stats to {fpath_stats}")


if __name__ == "__main__":
    get_statistics()
