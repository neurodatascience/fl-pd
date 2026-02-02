#!/usr/bin/env python

from pathlib import Path
from typing import Tuple

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
    required=True,
)
@click.option(
    "--output",
    "dpath_out",
    type=Path,
    required=True,
    envvar="DPATH_FL_DATA_LATEST_MEGA",
)
@click.option("--random-state", type=int, envvar="RANDOM_SEED")
@click.option("--n-splits", type=int, default=10)
def get_mega(
    target,
    dataset_tags_and_paths: Tuple[str, Path],
    dpath_out: Path,
    random_state: int = 3791,
    n_splits: int = 10,
):
    dfs = []
    for _, dataset_path in dataset_tags_and_paths:
        for i_split in range(n_splits):
            dataset = training_plan_factory(target).dataset_factory(
                target=target,
                i_split=i_split,
                random_state=random_state,
            )
            dataset.path = str(dataset_path)
            dataset.read()

            # output of data retriever, filtered by session, before any other transforms
            df = dataset.df_before_transforms.loc[dataset.idx]
            df.loc[:, "i_split"] = i_split

            dfs.append(df)

    df_mega = pd.concat(dfs, axis="index")

    dpath_out.mkdir(parents=True, exist_ok=True)
    fpath_mega = (
        dpath_out
        / f"mega_{'_'.join(sorted([tag for tag, _ in dataset_tags_and_paths]))}-{target}.tsv"
    )
    df_mega.to_csv(fpath_mega, sep="\t", index=True, header=True)
    print(f"Saved to {fpath_mega}")


if __name__ == "__main__":
    get_mega()
