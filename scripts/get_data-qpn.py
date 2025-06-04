#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS, COLS_PHENO
from fl_pd.utils.freesurfer import fs7_aparc_to_keep, fs7_aseg_to_keep

VISIT_IDS_ORDERED = [
    "legacy-moca",
    "Baseline (Arm 1: C-OPN)",
    "12 Months Follow-Up/Suivi (Arm 1: C-OPN)",
    "18 Months Follow-Up/Suivi (Arm 1: C-OPN)",
]


def get_df_pheno(fpath_demographics, fpath_age, fpath_diagnosis, fpath_moca):

    df_demographics = pd.read_csv(fpath_demographics).set_index("participant_id")
    df_age = (
        pd.read_csv(fpath_age).query('session == "ses-01"').set_index("participant_id")
    )
    df_diagnosis = pd.read_csv(fpath_diagnosis).set_index("participant_id")

    df_pheno = df_demographics.merge(
        df_age["MRI_age"], left_index=True, right_index=True
    )
    df_pheno = df_pheno.merge(
        df_diagnosis["diagnosis_group_for_analysis"], left_index=True, right_index=True
    )

    # rename columns
    df_pheno = df_pheno.rename(
        columns={
            "sex": "SEX",
            "MRI_age": "AGE",
            "education_legacy": "EDUCATION",
        }
    )

    # recode sex
    df_pheno["SEX"] = (
        df_pheno["SEX"]
        .map({"Male/Masculin": 1, "Female/Féminin": 0})
        .astype("category")
    )

    # recode diagnosis
    df_pheno["DIAGNOSIS"] = (
        df_pheno["diagnosis_group_for_analysis"]
        .map({"control": 0, "PD": 1})
        .astype("category")
    )

    # mark healthy controls
    df_pheno["IS_CONTROL"] = (
        df_pheno["diagnosis_group_for_analysis"]
        .map({"control": True, "PD": False})
        .astype("category")
    )

    # education
    # probably not best to have this as category
    # but the edge values are 6- and 20+ so this is not entirely numeric
    df_pheno["EDUCATION"] = df_pheno["EDUCATION"].astype("category")

    df_moca = pd.read_csv(fpath_moca).set_index("participant_id").sort_index()
    df_moca = df_moca.rename(columns={"MoCA Total Score": "MOCA"})
    df_moca = df_moca.sort_values(
        by="redcap_event_name",
        key=lambda x: x.map(VISIT_IDS_ORDERED.index),
    )

    df_moca_diff = pd.DataFrame(index=df_pheno.index)
    for participant_id in df_moca.index.unique():
        df_moca_participant = df_moca.loc[participant_id]
        if not isinstance(df_moca_participant, pd.Series):
            moca_scores = df_moca_participant["MOCA"].dropna()
            moca_diff = moca_scores.iloc[1:] - moca_scores.iloc[0]

            df_moca_diff.loc[participant_id, "MOCA_DIFF"] = moca_diff.min()

    df_pheno = df_pheno.merge(df_moca_diff, left_index=True, right_index=True)
    df_pheno["COG_DECLINE"] = df_pheno["MOCA_DIFF"].apply(
        lambda x: x <= -3 if not np.isnan(x) else np.nan
    )

    df_pheno = df_pheno[COLS_PHENO]
    print(f"Using {df_pheno.shape[1]} phenotypic features")

    return df_pheno


def get_df_imaging(fpath_aseg, fpath_aparc):

    df_aseg = pd.read_csv(fpath_aseg, sep="\t", dtype={"participant_id": str})
    df_aseg = df_aseg.set_index(["participant_id", "session_id"])
    df_aseg = fs7_aseg_to_keep(df_aseg)
    print(f"Using {df_aseg.shape[1]} aseg features")

    df_aparc = pd.read_csv(fpath_aparc, sep="\t", dtype={"participant_id": str})
    df_aparc = df_aparc.set_index(["participant_id", "session_id"])
    df_aparc = fs7_aparc_to_keep(df_aparc)
    print(f"Using {df_aparc.shape[1]} aparc features")

    df_imaging = pd.concat([df_aseg, df_aparc], axis="columns")

    df_imaging = df_imaging.query("session_id == 1").reset_index(
        "session_id", drop=True
    )

    return df_imaging


def get_df_qpn(
    fpath_demographics, fpath_age, fpath_diagnosis, fpath_moca, fpath_aseg, fpath_aparc
):
    df_pheno = get_df_pheno(
        fpath_demographics=fpath_demographics,
        fpath_age=fpath_age,
        fpath_diagnosis=fpath_diagnosis,
        fpath_moca=fpath_moca,
    )
    df_imaging = get_df_imaging(fpath_aseg, fpath_aparc)
    df_merged = df_pheno.join(df_imaging)

    return df_merged


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_demographics",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_DEMOGRAPHICS",
)
@click.argument(
    "fpath_age",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_AGE",
)
@click.argument(
    "fpath_diagnosis",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_DIAGNOSIS",
)
@click.argument(
    "fpath_moca",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_MOCA",
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
def get_data_qpn(
    fpath_demographics: Path,
    fpath_age: Path,
    fpath_diagnosis: Path,
    fpath_moca: Path,
    fpath_aseg: Path,
    fpath_aparc: Path,
    dpath_out: Path,
):
    fname_data_out_components = ["qpn"]
    dpath_out.mkdir(exist_ok=True)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_qpn = get_df_qpn(
        fpath_demographics=fpath_demographics,
        fpath_age=fpath_age,
        fpath_diagnosis=fpath_diagnosis,
        fpath_moca=fpath_moca,
        fpath_aseg=fpath_aseg,
        fpath_aparc=fpath_aparc,
    )
    if df_qpn.empty:
        raise ValueError("Empty dataset")
    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_qpn.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_qpn.shape} to {fpath_data_out}")


if __name__ == "__main__":
    get_data_qpn()
