import datetime
import os
from contextlib import contextmanager
from pathlib import Path


import numpy as np
import pandas as pd
from skrub import TableVectorizer

from fl_pd.utils.constants import MlSetup, DNAME_LATEST, DATE_FORMAT


def get_dpath_latest(dpath_parent, use_today=False):
    dpath_parent = Path(dpath_parent)
    dpath_latest = dpath_parent / DNAME_LATEST

    dpath_today = dpath_parent / datetime.datetime.today().strftime(DATE_FORMAT)
    if dpath_latest.exists():
        if use_today and dpath_latest.resolve() != dpath_today.resolve():
            if dpath_latest.is_symlink():
                dpath_latest.unlink()
            else:
                raise RuntimeError(f"{dpath_latest=} exists but is not a symlink")

    if not dpath_latest.exists():
        if dpath_latest.is_symlink():
            dpath_latest.unlink()
        dpath_today.mkdir(parents=True, exist_ok=True)
        dpath_latest.symlink_to(dpath_today, target_is_directory=True)

    return dpath_latest.resolve()


def load_Xy(
    fpath,
    target_cols,
    return_vectorizer=False,
    dataset=None,
    setup=None,
    datasets=None,
    null=False,
):
    is_mega = setup == MlSetup.MEGA

    df: pd.DataFrame = pd.read_csv(fpath, sep="\t")
    df = df.set_index("participant_id")
    df = df.dropna(axis="index", how="any")

    table_vectorizer = TableVectorizer()
    df = table_vectorizer.fit_transform(df)

    if is_mega and not (any(["dataset_" in col for col in df.columns])):
        if datasets is None:
            raise ValueError("datasets is None")
        for col_dataset in datasets:
            if col_dataset not in df.columns:
                df.loc[:, f"dataset_{col_dataset}"] = 1 if col_dataset == dataset else 0

    df = df[sorted(df.columns)]

    y = df.loc[:, target_cols]
    X = df.drop(columns=target_cols)

    if null:
        rng = np.random.default_rng()
        idx = np.arange(len(y))
        rng.shuffle(idx)
        y = y.iloc[idx]

    if return_vectorizer:
        return X, y, table_vectorizer
    else:
        return X, y


@contextmanager
def working_directory(dpath):
    dpath_old = Path.cwd()
    os.chdir(dpath)
    try:
        yield
    finally:
        os.chdir(dpath_old)
