import numpy as np
import pandas as pd
from skrub import TableVectorizer

from fl_pd.utils.constants import MlSetup


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
