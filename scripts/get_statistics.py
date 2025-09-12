#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument(
    "datasets",
    nargs=-1,
)
@click.option("--tag", type=str, help="Data subdirectory name")
@click.option("--suffix", type=str, help="Train/test partition suffix")
def get_statistics(dpath_data, datasets, tag, suffix):
    for dataset in datasets:
        tag_with_dataset = f"{dataset}-{tag}"
        df = pd.read_csv(
            dpath_data / tag_with_dataset / f"{tag_with_dataset}{suffix}.tsv",
            index_col="participant_id",
            sep="\t",
        )

        df_stats = df.describe()
        fpath_stats = (
            dpath_data / tag_with_dataset / f"{tag_with_dataset}{suffix}-stats.tsv"
        )
        df_stats.to_csv(fpath_stats, sep="\t", index=True, header=True)
        print(f"Saved stats to {fpath_stats}")


if __name__ == "__main__":
    get_statistics()
