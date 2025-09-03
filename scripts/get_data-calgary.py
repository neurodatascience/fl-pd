#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.io import get_dpath_latest
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS, COLS_PHENO
from fl_pd.utils.freesurfer import fs6_to_fs7, fs7_aparc_to_keep, fs7_aseg_to_keep
from fl_pd.pheno import cog_decline_from_moca_rate


def get_df_pheno(fpath_pheno):

    df_pheno = pd.read_csv(
        fpath_pheno, sep="\t", low_memory=False, dtype={"participant_id": str}
    )
    df_pheno = df_pheno.set_index("participant_id")

    df_pheno["date_neuropsych"] = pd.to_datetime(df_pheno["date_neuropsych"])

    df_pheno["AGE"] = df_pheno["age"]

    # use PPMI coding for sex
    df_pheno["SEX"] = df_pheno["sex"].map({"Male": 1, "Female": 0})

    # recode diagnosis
    df_pheno["DIAGNOSIS"] = df_pheno["diagnosis"].map(
        lambda x: 0 if x == "Non-PD" else (1 if not pd.isna(x) else pd.NA)
    )

    df_pheno["IS_CONTROL"] = df_pheno["diagnosis"].map(
        lambda x: True if x == "Non-PD" else (False if not pd.isna(x) else pd.NA)
    )
    df_pheno["IS_CASE"] = df_pheno["IS_CONTROL"].apply(
        lambda x: not x if pd.notna(x) else x
    )
    for col in ("IS_CONTROL", "IS_CASE", "DIAGNOSIS", "SEX"):
        df_pheno[col] = df_pheno[col].astype("category")

    data_for_df_moca_rate = []
    for participant_id in df_pheno.index.get_level_values("participant_id").unique():
        df_participant = df_pheno.loc[[participant_id]]
        df_participant["moca_years"] = (
            df_participant["date_neuropsych"]
            - df_participant.query("visit_id == 'TP1'")["date_neuropsych"].item()
        ).dt.days / 365.25

        moca_values_and_years = df_participant[["moca", "moca_years"]].dropna(how="any")

        moca_values = moca_values_and_years["moca"]
        moca_years = moca_values_and_years["moca_years"]

        # skip if only single timepoint for MoCA
        if len(moca_years) < 2 or moca_years.max() < 0.5:
            continue

        moca_rate = np.polyfit(moca_years, moca_values, 1)[0]

        data_for_df_moca_rate.append(
            {
                "participant_id": participant_id,
                "MOCA_RATE": moca_rate,
            }
        )

    df_moca_rate = pd.DataFrame(data_for_df_moca_rate)
    df_pheno = df_pheno.merge(
        df_moca_rate.set_index("participant_id"),
        left_index=True,
        right_index=True,
        how="left",
    )
    df_pheno["COG_DECLINE"] = df_pheno["MOCA_RATE"].apply(cog_decline_from_moca_rate)

    df_pheno = df_pheno.query('visit_id == "TP1"')

    df_pheno = df_pheno.loc[:, COLS_PHENO]

    print(f"Using {df_pheno.shape[1]} phenotypic features")

    return df_pheno


def get_df_imaging(
    fpath_aseg,
    fpath_aparc,
):
    df_aseg = pd.read_csv(fpath_aseg, sep="\t", dtype={"participant_id": str})
    df_aseg = df_aseg.set_index(["participant_id", "session_id"])
    df_aseg = fs6_to_fs7(df_aseg)
    df_aseg = fs7_aseg_to_keep(df_aseg)
    print(f"Using {df_aseg.shape[1]} aseg features")

    df_aparc = pd.read_csv(fpath_aparc, sep="\t", dtype={"participant_id": str})
    df_aparc = df_aparc.set_index(["participant_id", "session_id"])
    df_aparc = fs6_to_fs7(df_aparc)
    df_aparc = fs7_aparc_to_keep(df_aparc)
    print(f"Using {df_aparc.shape[1]} aparc features")

    df_aparc = pd.concat([df_aseg, df_aparc], axis="columns")
    df_aparc = df_aparc.query('session_id == "TP1"').reset_index(
        "session_id", drop=True
    )

    return df_aparc


def get_df_calgary(fpath_pheno, fpath_aseg, fpath_aparc):
    df_pheno = get_df_pheno(fpath_pheno)
    df_imaging = get_df_imaging(fpath_aseg, fpath_aparc)
    df_merged = df_pheno.join(df_imaging, how="inner")
    df_merged = df_merged.sort_index()
    return df_merged


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_pheno",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_CALGARY_PHENO",
)
@click.argument(
    "fpath_aseg",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_CALGARY_ASEG",
)
@click.argument(
    "fpath_aparc",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_CALGARY_APARC",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_FL_DATA",
)
def get_data_calgary(
    fpath_pheno: Path, fpath_aseg: Path, fpath_aparc: Path, dpath_out: Path
):

    fname_data_out_components = ["calgary"]
    dpath_out = get_dpath_latest(dpath_out)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_calgary = get_df_calgary(
        fpath_pheno=fpath_pheno,
        fpath_aseg=fpath_aseg,
        fpath_aparc=fpath_aparc,
    )
    if df_calgary.empty:
        raise ValueError("Empty dataset")

    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_calgary.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_calgary.shape} to {fpath_data_out}")


if __name__ == "__main__":

    get_data_calgary()
