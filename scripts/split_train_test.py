#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fl_pd.ml_spec import get_target_from_tag
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS

DEFAULT_MIN_AGE = None


def _standardize_features(df, target_col, standardizer=None):
    df_X = df.drop(columns=target_col)
    if standardizer is None:
        standardizer = StandardScaler().fit(df_X)
    df_X_standardized = pd.DataFrame(
        standardizer.transform(df_X),
        index=df_X.index,
        columns=standardizer.get_feature_names_out(),
    )
    df_standardized = pd.concat([df_X_standardized, df[[target_col]]], axis="columns")
    return (
        df_standardized.loc[:, df.columns],
        standardizer,
    )


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
    "--standardize/--no-standardize", default=True, help="Standardize data after split"
)
@click.option("--min-age", type=int, default=DEFAULT_MIN_AGE)
@click.option(
    "--random-state",
    type=int,
    envvar="RANDOM_SEED",
    help="Random state for reproducibility",
)
def split_train_test(
    dpath_data: Path,
    tags: list[str],
    n_splits,
    stratify_col,
    shuffle,
    standardize,
    min_age,
    random_state,
):

    for tag in tags:
        print(f"===== {tag} =====")

        fpath_data = dpath_data / tag / f"{tag}.tsv"

        df: pd.DataFrame = pd.read_csv(fpath_data, sep="\t", index_col="participant_id")
        print(f"Full dataframe shape: {df.shape}")

        data_tag = fpath_data.stem
        if min_age:
            df = df.query(f"AGE >= {min_age}")
            data_tag = f"{data_tag}-{min_age}"
        if standardize:
            data_tag = f"{data_tag}-standardized"

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

            if standardize:
                target_col = get_target_from_tag(data_tag).value.upper()
                df_train, standardizer = _standardize_features(df_train, target_col)
                df_test, _ = _standardize_features(
                    df_test, target_col, standardizer=standardizer
                )

            dpath_out = fpath_data.parent.parent / data_tag
            dpath_out.mkdir(exist_ok=True)
            fpath_train = dpath_out / f"{data_tag}-{i_split}train.tsv"
            fpath_test = dpath_out / f"{data_tag}-{i_split}test.tsv"

            df_train.to_csv(fpath_train, sep="\t", index=True)
            df_test.to_csv(fpath_test, sep="\t", index=True)
            print(f"\tTrain {df_train.shape} -> {fpath_train}")
            print(f"\tTest {df_test.shape} -> {fpath_test}")


if __name__ == "__main__":
    split_train_test()
