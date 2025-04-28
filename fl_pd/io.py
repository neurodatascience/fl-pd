import pandas as pd
from skrub import TableVectorizer


def load_Xy(
    fpath,
    target_cols,
    min_age=None,
    return_vectorizer=False,
    dataset=None,
    setup=None,
    datasets=None,
):
    is_mega = setup == "mega"

    df: pd.DataFrame = pd.read_csv(fpath, sep="\t")
    df = df.set_index("participant_id")
    df = df.dropna(axis="index", how="any")

    if min_age is not None:
        df = df.query(f"AGE >= {min_age}")

    table_vectorizer = TableVectorizer()
    df = table_vectorizer.fit_transform(df)

    if is_mega and dataset not in df.columns:
        if datasets is None:
            raise ValueError("datasets is None")
        for col_dataset in datasets:
            if col_dataset not in df.columns:
                df.loc[:, col_dataset] = 1 if col_dataset == dataset else 0

    y = df.loc[:, target_cols]
    X = df.drop(columns=target_cols)

    if return_vectorizer:
        return X, y, table_vectorizer
    else:
        return X, y
