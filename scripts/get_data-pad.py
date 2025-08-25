#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_pd.io import get_dpath_latest
from fl_pd.pheno import cog_decline_from_pad_mci
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    COLS_PHENO,
    DPATH_RELATIVE_PAD_IMAGING_SESSIONS,
)
from fl_pd.utils.freesurfer import fs7_aparc_to_keep, fs7_aseg_to_keep


def get_df_pheno(fpath_demographics, fpath_age, fpath_mci) -> pd.DataFrame:

    # sex and other demographics (no age)
    df_demographics = pd.read_csv(
        fpath_demographics,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index("participant_id")
    df_demographics["SEX"] = df_demographics["Sex"].map({"Female": 0, "Male": 1})

    # age at imaging visits
    df_age = pd.read_csv(
        fpath_age,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index(["participant_id", "session_id"])
    df_age["AGE"] = df_age["age"].apply(lambda x: x / 12)

    df_mci = pd.read_csv(
        fpath_mci,
        sep="\t",
        dtype={"participant_id": str},
    ).set_index("participant_id")

    df_cog_decline = cog_decline_from_pad_mci(df_mci)

    df_pheno = df_age.copy()
    df_pheno = df_pheno.merge(
        df_demographics.loc[:, "SEX"], left_index=True, right_index=True
    )
    df_pheno = df_pheno.merge(
        df_cog_decline, left_index=True, right_index=True, how="left"
    )

    df_pheno["DIAGNOSIS"] = 0

    # a bit weird but this is because everyone is prodromal (no clinical diagnosis)
    # this way the cog. decline classification task uses everyone
    # and brain age only uses non-decliners
    df_pheno["IS_CONTROL"] = ~df_pheno["COG_DECLINE"]
    df_pheno["IS_CASE"] = True

    for col in ["COG_DECLINE", "SEX", "DIAGNOSIS", "IS_CONTROL", "IS_CASE"]:
        df_pheno[col] = df_pheno[col].astype("category")

    df_pheno = df_pheno[COLS_PHENO]
    print(f"Using {df_pheno.shape[1]} phenotypic features")
    return df_pheno


def get_df_imaging(fpath_aseg, fpath_aparc, fpath_imaging_sessions) -> pd.DataFrame:

    participants_sessions = list(
        pd.read_csv(
            fpath_imaging_sessions,
            sep="\t",
            header=None,
            dtype=str,
        ).itertuples(index=False, name=None)
    )

    df_aseg = pd.read_csv(fpath_aseg, sep="\t", dtype={"participant_id": str})
    df_aseg = df_aseg.set_index(["participant_id", "session_id"])

    df_aparc = pd.read_csv(fpath_aparc, sep="\t", dtype={"participant_id": str})
    df_aparc = df_aparc.set_index(["participant_id", "session_id"])

    df_aseg = fs7_aseg_to_keep(df_aseg)
    df_aparc = fs7_aparc_to_keep(df_aparc)

    print(f"Using {df_aseg.shape[1]} aseg features")
    print(f"Using {df_aparc.shape[1]} aparc features")

    df_imaging = pd.concat([df_aseg, df_aparc], axis="columns")

    df_imaging = df_imaging.loc[participants_sessions]
    return df_imaging


def get_df_pad(
    fpath_demographics,
    fpath_age,
    fpath_mci,
    fpath_aseg,
    fpath_aparc,
    fpath_imaging_sessions,
) -> pd.DataFrame:

    df_pheno = get_df_pheno(fpath_demographics, fpath_age, fpath_mci)
    df_imaging = get_df_imaging(fpath_aseg, fpath_aparc, fpath_imaging_sessions)

    # merge
    df_merged = df_pheno.join(df_imaging, how="inner")
    df_merged = df_merged.reset_index("session_id", drop=True)

    return df_merged


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
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
    "fpath_aseg",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_ASEG",
)
@click.argument(
    "fpath_aparc",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_APARC",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_FL_DATA",
)
def get_data_pad(
    fpath_demographics: Path,
    fpath_age: Path,
    fpath_mci: Path,
    fpath_aseg: Path,
    fpath_aparc: Path,
    dpath_out: Path,
):
    fname_data_out_components = ["pad"]

    dpath_out = get_dpath_latest(dpath_out)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    fpath_imaging_sessions = dpath_out / DPATH_RELATIVE_PAD_IMAGING_SESSIONS

    df_ppmi = get_df_pad(
        fpath_demographics=fpath_demographics,
        fpath_age=fpath_age,
        fpath_mci=fpath_mci,
        fpath_aseg=fpath_aseg,
        fpath_aparc=fpath_aparc,
        fpath_imaging_sessions=fpath_imaging_sessions,
    )
    if df_ppmi.empty:
        raise ValueError("Empty dataset")

    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_ppmi.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_ppmi.shape} to {fpath_data_out}")


if __name__ == "__main__":
    get_data_pad()
