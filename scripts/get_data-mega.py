#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd


@click.command()
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument("datasets", type=click.Choice(["adni", "ppmi", "qpn"]), nargs=-1)
@click.option("--tag", type=str)
def get_data_mega(dpath_data, datasets, tag):
    dfs = {}
    for dataset in datasets:
        df = pd.read_csv(
            dpath_data / f"{dataset}-{tag}.tsv",
            index_col="participant_id",
            sep="\t",
        )
        print(f"{dataset}:\t{df.shape}")
        dfs[dataset] = df
    df_mega = pd.concat(dfs, names=["dataset"])
    print(f"mega:\t{df_mega.shape}")

    datasets_str = "_".join(datasets)
    fpath_mega = dpath_data / f"mega_{datasets_str}-{tag}.tsv"
    df_mega.to_csv(fpath_mega, sep="\t", index=True, header=True)
    print(f"Saved to {fpath_mega}")


if __name__ == "__main__":
    get_data_mega()
