#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument(
    "tags",
    type=str,
    nargs=-1,
)
@click.option(
    "--n-splits",
    type=click.IntRange(min=2),
    default=10,
    help="Number of train-test splits",
)
@click.option(
    "--stratify-col",
    type=str,
    help="Column to stratify on (default: no stratification)",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="Shuffle data before splitting",
)
@click.option(
    "--random-state",
    type=int,
    envvar="RANDOM_SEED",
    help="Random state for reproducibility",
)
def split_train_test(
    dpath_data: Path, tags: list[str], n_splits, stratify_col, shuffle, random_state
):

    for tag in tags:
        print(f"===== {tag} =====")

        fpath_data = dpath_data / tag / f"{tag}.tsv"

        df: pd.DataFrame = pd.read_csv(fpath_data, sep="\t")
        print(f"Full dataframe shape: {df.shape}")

        if stratify_col == "DECLINE":
            stratify = df["DECLINE"]
        elif stratify_col == "AGE":
            age_bins = np.arange(0, 100, 10)
            stratify = pd.cut(df["AGE"], bins=age_bins).astype(str)
        else:
            stratify = np.zeros(df.shape[0])

        cv_splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        for i_split, (idx_train, idx_test) in enumerate(
            cv_splitter.split(X=df, y=stratify),
        ):
            print(f"Split {i_split}")
            df_train: pd.DataFrame = df.iloc[idx_train]
            df_test: pd.DataFrame = df.iloc[idx_test]

            fpath_train = fpath_data.with_name(f"{fpath_data.stem}-{i_split}train.tsv")
            fpath_test = fpath_data.with_name(f"{fpath_data.stem}-{i_split}test.tsv")

            df_train.to_csv(fpath_train, sep="\t", index=False)
            df_test.to_csv(fpath_test, sep="\t", index=False)
            print(f"\tTrain {df_train.shape} -> {fpath_train}")
            print(f"\tTest {df_test.shape} -> {fpath_test}")


if __name__ == "__main__":
    split_train_test()
