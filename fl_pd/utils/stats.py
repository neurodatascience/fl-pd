import pandas as pd


def standardize_df(df: pd.DataFrame, fpath_stats, cols_to_ignore=None) -> pd.DataFrame:
    if cols_to_ignore is None:
        cols_to_ignore = []
    cols_to_ignore.append("dataset")
    df_stats = pd.read_csv(fpath_stats, sep="\t", index_col=0)
    for col in df.columns:
        if col in cols_to_ignore:
            continue
        if col not in df_stats.columns:
            raise ValueError(f"{col=} not in {fpath_stats=}")
        mean = df_stats.at["mean", col]
        std = df_stats.at["std", col]
        if std == 0:
            raise ValueError(f"std is zero for {col=} in {fpath_stats=}")
        df[col] = (df[col] - mean) / std
    return df
