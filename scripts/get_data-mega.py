#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from utils import CLICK_CONTEXT_SETTINGS


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument("datasets", type=click.Choice(["adni", "ppmi", "qpn"]), nargs=-1)
@click.option("--tag", type=str, help="Data subdirectory name")
@click.option("--suffix", type=str, help="Train/test partition suffix")
@click.option("--shuffle/--no-shuffle", default=True)
@click.option("--random-state", type=int, envvar="RANDOM_SEED")
def get_data_mega(dpath_data, datasets, tag, suffix, shuffle, random_state):
    dfs = {}
    for dataset in datasets:
        tag_with_dataset = f"{dataset}-{tag}"
        df = pd.read_csv(
            dpath_data / tag_with_dataset / f"{tag_with_dataset}{suffix}.tsv",
            index_col="participant_id",
            sep="\t",
        )
        print(f"{dataset}:\t{df.shape}")
        dfs[dataset] = df
    df_mega = pd.concat(dfs, names=["dataset"])
    if shuffle:
        df_mega = df_mega.sample(frac=1, replace=False, random_state=random_state)
    print(f"mega:\t{df_mega.shape}")

    tag_output = "_".join(["mega"] + list(datasets)) + f"-{tag}"
    fpath_mega = dpath_data / tag_output / f"{tag_output}{suffix}.tsv"
    fpath_mega.parent.mkdir(parents=True, exist_ok=True)
    df_mega.to_csv(fpath_mega, sep="\t", index=True, header=True)
    print(f"Saved to {fpath_mega}")


if __name__ == "__main__":
    get_data_mega()
