#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.io import get_dpath_latest
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS, COLS_PHENO
from fl_pd.utils.freesurfer import fs6_to_fs7, fs7_aparc_to_keep, fs7_aseg_to_keep
from fl_pd.pheno import cog_decline_from_mmse_rate


def get_df_pheno(fpath_pheno):

    df_pheno = pd.read_csv(fpath_pheno, low_memory=False)
    df_pheno["participant_id"] = "ADNI" + df_pheno["PTID"].str.replace("_", "")
    df_pheno = df_pheno.set_index("participant_id")

    df_pheno["EXAMDATE"] = pd.to_datetime(df_pheno["EXAMDATE"])
    df_pheno = df_pheno.sort_values(by="EXAMDATE", ascending=True)

    # use PPMI coding for sex
    df_pheno["SEX"] = df_pheno["PTGENDER"].map({"Male": 1, "Female": 0})

    # recode diagnosis
    df_pheno["DIAGNOSIS"] = df_pheno["DX_bl"].map(
        lambda x: 0 if x == "CN" else (1 if not pd.isna(x) else pd.NA)
    )

    df_pheno["IS_CONTROL"] = df_pheno["DX_bl"].map(
        lambda x: True if x == "CN" else (False if not pd.isna(x) else pd.NA)
    )
    df_pheno["IS_CASE"] = df_pheno["IS_CONTROL"].apply(
        lambda x: not x if pd.notna(x) else x
    )
    for col in ("IS_CONTROL", "IS_CASE", "DIAGNOSIS", "SEX"):
        df_pheno[col] = df_pheno[col].astype("category")

    # visits = [
    #     "bl",
    #     "m0",
    #     "m03",
    #     "m06",
    #     "m12",
    #     "m18",
    #     "m24",
    #     "m30",
    #     "m36",
    #     "m42",
    #     "m48",
    #     "m54",
    #     "m60",
    # ]
    df_pheno_5_years = df_pheno  # .query("VISCODE in @visits")
    data_for_df_mmse_rate = []
    for participant_id in df_pheno_5_years.index.get_level_values(
        "participant_id"
    ).unique():
        df_participant = df_pheno_5_years.loc[[participant_id]]

        mmse_values_and_years = df_participant[["MMSE", "Years_bl"]].dropna(how="any")

        mmse_values = mmse_values_and_years["MMSE"]
        mmse_years = mmse_values_and_years["Years_bl"]

        # skip if only single timepoint for MMSE
        if len(mmse_years) < 2 or mmse_years.max() < 0.5:
            continue

        mmse_rate = np.polyfit(mmse_years, mmse_values, 1)[0]

        data_for_df_mmse_rate.append(
            {
                "participant_id": participant_id,
                "MMSE_RATE": mmse_rate,
            }
        )

    df_mmse_rate = pd.DataFrame(data_for_df_mmse_rate)
    df_pheno = df_pheno.merge(
        df_mmse_rate.set_index("participant_id"),
        left_index=True,
        right_index=True,
        how="left",
    )
    df_pheno["COG_DECLINE"] = df_pheno["MMSE_RATE"].apply(cog_decline_from_mmse_rate)

    df_pheno = df_pheno.query('VISCODE == "bl"')

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
    df_aparc = df_aparc.query('session_id == "bl"').reset_index("session_id", drop=True)

    return df_aparc


def get_df_adni(fpath_pheno, fpath_aseg, fpath_aparc):
    df_pheno = get_df_pheno(fpath_pheno)
    df_imaging = get_df_imaging(fpath_aseg, fpath_aparc)
    df_merged = df_pheno.join(df_imaging, how="inner")
    df_merged = df_merged.sort_index()
    return df_merged


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_pheno",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_ADNI_PHENO",
)
@click.argument(
    "fpath_aseg",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_ADNI_ASEG",
)
@click.argument(
    "fpath_aparc",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_ADNI_APARC",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_FL_DATA",
)
def get_data_adni(
    fpath_pheno: Path, fpath_aseg: Path, fpath_aparc: Path, dpath_out: Path
):

    fname_data_out_components = ["adni"]
    dpath_out = get_dpath_latest(dpath_out)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_adni = get_df_adni(
        fpath_pheno=fpath_pheno,
        fpath_aseg=fpath_aseg,
        fpath_aparc=fpath_aparc,
    )
    if df_adni.empty:
        raise ValueError("Empty dataset")

    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_adni.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_adni.shape} to {fpath_data_out}")


if __name__ == "__main__":

    get_data_adni()
