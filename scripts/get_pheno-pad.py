#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.pheno import cog_decline_from_pad_mci
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS


def get_df_pheno(
    fpath_manifest, fpath_demographics, fpath_age, fpath_mci
) -> pd.DataFrame:

    df_manifest = pd.read_csv(
        fpath_manifest,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index(["participant_id", "session_id"])

    # sex and other demographics (no age)
    df_demographics = pd.read_csv(
        fpath_demographics,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index("participant_id")

    # age at imaging visits
    df_age = pd.read_csv(
        fpath_age,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index(["participant_id", "session_id"])
    df_age = df_manifest.loc[:, []].merge(
        df_age, left_index=True, right_index=True, how="outer"
    )
    df_age["months"] = (
        df_age.index.get_level_values("session_id")
        .str.extract(r"(\d+)")
        .astype(int)
        .to_numpy()
    )
    dfs_participant_id = []
    for participant_id in df_age.index.get_level_values("participant_id").unique():
        df_age_participant = df_age.loc[[participant_id]].sort_values(["months", "age"])
        idx_na = df_age_participant["age"].isna()
        if not idx_na.all():
            latest_available_age = df_age_participant.dropna(
                subset=["age"], axis="index"
            ).iloc[[-1]]
            df_age_participant.loc[idx_na, "age"] = (
                df_age_participant.loc[idx_na, "months"]
                + latest_available_age["age"].item()
                - latest_available_age["months"].item()
            )
        dfs_participant_id.append(df_age_participant)
    df_age = pd.concat(dfs_participant_id, axis="index")
    df_age["age"] = df_age["age"].apply(lambda x: np.round(x / 12, 1))

    df_mci = pd.read_csv(
        fpath_mci,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index("participant_id")

    df_cog_decline = cog_decline_from_pad_mci(df_mci)

    df_pheno = df_age.copy()
    df_pheno = df_pheno.merge(
        df_demographics.loc[:, "Sex"], left_index=True, right_index=True, how="outer"
    )
    df_pheno = df_pheno.merge(
        df_cog_decline, left_index=True, right_index=True, how="outer"
    )

    for col in ["cog_decline", "Sex"]:
        df_pheno[col] = df_pheno[col].astype("category")

    df_pheno = df_pheno.drop(columns=["n_months_switch", "months"])

    return df_pheno


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_MANIFEST",
)
@click.argument(
    "fpath_demographics",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_DEMOGRAPHICS",
)
@click.argument(
    "fpath_age",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_AGE",
)
@click.argument(
    "fpath_mci",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_MCI",
)
@click.argument(
    "fpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="FPATH_PAD_PHENO_CLEAN",
)
def get_data_pad(
    fpath_manifest: Path,
    fpath_demographics: Path,
    fpath_age: Path,
    fpath_mci: Path,
    fpath_out: Path,
):
    df = get_df_pheno(fpath_manifest, fpath_demographics, fpath_age, fpath_mci)
    if df.empty:
        raise ValueError("Empty dataset")

    fpath_out.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(fpath_out, sep="\t")
    print(f"Saved final dataframe of shape {df.shape} to {fpath_out}")


if __name__ == "__main__":
    get_data_pad()
