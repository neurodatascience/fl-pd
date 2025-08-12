#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.io import get_dpath_latest
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS, COLS_PHENO
from fl_pd.utils.freesurfer import fs6_to_fs7, fs7_aparc_to_keep, fs7_aseg_to_keep
from fl_pd.pheno import cog_decline_from_moca_rate

VISIT_IDS_ORDERED = ["SC", "BL", "V04", "V06", "V08", "V10"]


def get_df_pheno(fpath_pheno) -> pd.DataFrame:

    # load
    df_pheno = pd.read_csv(
        fpath_pheno,
        sep="\t",
        dtype={"participant_id": str},
    )

    # set/sort index
    df_pheno = df_pheno.set_index(["participant_id", "visit_id"])
    df_pheno = df_pheno.sort_index(
        key=lambda x: x.map(
            lambda y: VISIT_IDS_ORDERED.index(y) if y in VISIT_IDS_ORDERED else y
        )
    )

    # recode diagnosis
    df_pheno["DIAGNOSIS"] = (
        df_pheno["COHORT_DEFINITION"]
        .map({"Healthy Control": 0, "Parkinson's Disease": 1, "Prodromal": 0})
        .astype("category")
    )

    # PD patients + prodromal are not control
    df_pheno["IS_CONTROL"] = (
        df_pheno["COHORT_DEFINITION"]
        .map(
            {"Healthy Control": True, "Parkinson's Disease": False, "Prodromal": False}
        )
        .astype("category")
    )

    # convert to datetime
    for col in ["DATE_OF_BIRTH", "DATE_OF_DIAGNOSIS", "DATE_MOCA"]:
        df_pheno[col] = pd.to_datetime(df_pheno[col], format="%m/%Y", errors="coerce")

    # convert to categorical
    for col in ["SEX", "COHORT_DEFINITION", "PRIMARY_DIAGNOSIS"]:
        df_pheno[col] = df_pheno[col].astype("category")

    # determine cognitive decline
    data_for_df_moca_rate = []
    for participant_id in df_pheno.index.get_level_values("participant_id").unique():
        df_participant = df_pheno.loc[participant_id]
        moca_values_and_dates = df_participant[["MOCA", "DATE_MOCA"]].dropna(how="any")

        moca_values = moca_values_and_dates["MOCA"]
        moca_dates_years = (
            moca_values_and_dates["DATE_MOCA"]
            - moca_values_and_dates["DATE_MOCA"].min()
        ).dt.days / 365.25

        # skip if only single timepoint for MoCA
        if len(moca_dates_years) < 2 or moca_dates_years.max() < 0.5:
            continue

        moca_rate = np.polyfit(moca_dates_years, moca_values, 1)[0]

        data_for_df_moca_rate.append(
            {
                "participant_id": participant_id,
                "MOCA_RATE": moca_rate,
            }
        )
    df_moca_rate = pd.DataFrame(data_for_df_moca_rate)
    df_pheno = (
        df_pheno.reset_index("visit_id")
        .merge(
            df_moca_rate.set_index("participant_id"),
            left_index=True,
            right_index=True,
            how="left",
        )
        .set_index("visit_id", append=True)
    )
    df_pheno["COG_DECLINE"] = df_pheno["MOCA_RATE"].apply(cog_decline_from_moca_rate)

    # keep only BL
    df_pheno = df_pheno.query('visit_id == "BL"').reset_index("visit_id", drop=True)

    df_pheno = df_pheno[COLS_PHENO]
    print(f"Using {df_pheno.shape[1]} phenotypic features")

    return df_pheno


def get_df_imaging(fpath_aseg, fpath_aparc, is_fs6=False) -> pd.DataFrame:

    df_aseg = pd.read_csv(fpath_aseg, sep="\t", dtype={"participant_id": str})
    df_aseg = df_aseg.set_index(["participant_id", "session_id"])

    df_aparc = pd.read_csv(fpath_aparc, sep="\t", dtype={"participant_id": str})
    df_aparc = df_aparc.set_index(["participant_id", "session_id"])

    if is_fs6:
        df_aseg = fs6_to_fs7(df_aseg)
        df_aparc = fs6_to_fs7(df_aparc)

    df_aseg = fs7_aseg_to_keep(df_aseg)
    df_aparc = fs7_aparc_to_keep(df_aparc)

    print(f"Using {df_aseg.shape[1]} aseg features")
    print(f"Using {df_aparc.shape[1]} aparc features")

    df_imaging = pd.concat([df_aseg, df_aparc], axis="columns")

    # keep only BL
    df_imaging = df_imaging.query('session_id == "BL"').reset_index(
        "session_id", drop=True
    )

    return df_imaging


def get_df_ppmi(fpath_pheno, fpath_aseg, fpath_aparc, is_fs6=False) -> pd.DataFrame:

    df_pheno = get_df_pheno(fpath_pheno)
    df_imaging = get_df_imaging(fpath_aseg, fpath_aparc, is_fs6=is_fs6)

    # merge
    df_merged = df_pheno.join(df_imaging, how="inner")

    return df_merged


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_pheno",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PPMI_PHENO",
)
@click.argument(
    "fpath_aseg",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PPMI_ASEG",
)
@click.argument(
    "fpath_aparc",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PPMI_APARC",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_FL_DATA",
)
@click.option("--fs6/--fs7", "use_fs6", default=False)
@click.option(
    "--fpath-aseg-fs6",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=False,
    envvar="FPATH_PPMI_ASEG_FS6",
)
@click.option(
    "--fpath-aparc-fs6",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=False,
    envvar="FPATH_PPMI_APARC_FS6",
)
def get_data_ppmi(
    fpath_pheno: Path,
    fpath_aseg: Path,
    fpath_aparc: Path,
    dpath_out: Path,
    fpath_aseg_fs6=None,
    fpath_aparc_fs6=None,
    use_fs6=False,
):

    if use_fs6 and (fpath_aseg_fs6 is None or fpath_aparc_fs6 is None):
        raise ValueError("Must provide path to FS6 imaging data if --fs6 is given")

    fname_data_out_components = ["ppmi"]

    if use_fs6:
        fpath_aseg = fpath_aseg_fs6
        fpath_aparc = fpath_aparc_fs6
        fname_data_out_components.append("fs6")

    dpath_out = get_dpath_latest(dpath_out)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_ppmi = get_df_ppmi(
        fpath_pheno=fpath_pheno,
        fpath_aseg=fpath_aseg,
        fpath_aparc=fpath_aparc,
        is_fs6=use_fs6,
    )
    if df_ppmi.empty:
        raise ValueError("Empty dataset")

    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_ppmi.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_ppmi.shape} to {fpath_data_out}")


if __name__ == "__main__":
    get_data_ppmi()
