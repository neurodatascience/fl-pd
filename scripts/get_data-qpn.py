#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_pd.utils.freesurfer import fs7_aparc_to_keep, fs7_aseg_to_keep

VISIT_IDS_ORDERED = [
    "legacy-moca",
    "Baseline (Arm 1: C-OPN)",
    "12 Months Follow-Up/Suivi (Arm 1: C-OPN)",
    "18 Months Follow-Up/Suivi (Arm 1: C-OPN)",
]


def get_df_pheno(
    fpath_demographics,
    fpath_age,
    fpath_diagnosis,
    fpath_moca,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_diag=False,
    include_cases=True,
    include_controls=False,
):
    cols = []
    if include_decline:
        cols.append("COG_DECLINE")
    if include_age:
        cols.append("AGE")
    if include_sex:
        cols.append("SEX")
    if include_diag:
        cols.append("DIAGNOSIS")

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

    diagnoses_to_include = []
    if include_cases:
        diagnoses_to_include.append("PD")
    if include_controls:
        diagnoses_to_include.append("control")
    df_pheno = df_pheno.query("diagnosis_group_for_analysis in @diagnoses_to_include")

    df_pheno = df_pheno[cols]
    print(f"Keeping phenotypic columns: {cols}")

    return df_pheno


def get_df_imaging(fpath_imaging, include_aparc=True, include_aseg=False):

    df_imaging = pd.read_csv(fpath_imaging, sep="\t", dtype={"participant_id": str})
    df_imaging = df_imaging.set_index(["participant_id", "session_id"])

    df_aparc = df_imaging.drop(
        columns=[col for col in df_imaging.columns if "thickness" not in col],
    )
    df_aparc = fs7_aparc_to_keep(df_aparc)
    df_aseg = df_imaging.drop(
        columns=[col for col in df_imaging.columns if "thickness" in col],
    )
    df_aseg = fs7_aseg_to_keep(df_aseg)

    dfs_imaging = []
    if include_aparc:
        dfs_imaging.append(df_aparc)
    if include_aseg:
        dfs_imaging.append(df_aseg)

    df_imaging = pd.concat(dfs_imaging, axis="columns")
    df_imaging = df_imaging.query("session_id == 1").reset_index(
        "session_id", drop=True
    )

    return df_imaging


def get_df_qpn(
    fpath_demographics,
    fpath_age,
    fpath_diagnosis,
    fpath_moca,
    fpath_imaging,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_diag=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
):
    df_pheno = get_df_pheno(
        fpath_demographics=fpath_demographics,
        fpath_age=fpath_age,
        fpath_diagnosis=fpath_diagnosis,
        fpath_moca=fpath_moca,
        include_decline=include_decline,
        include_age=include_age,
        include_sex=include_sex,
        include_diag=include_diag,
        include_cases=include_cases,
        include_controls=include_controls,
    )
    df_imaging = get_df_imaging(
        fpath_imaging, include_aparc=include_aparc, include_aseg=include_aseg
    )

    if include_aparc or include_aseg:
        df_merged = df_pheno.merge(df_imaging, left_index=True, right_index=True)
    else:
        df_merged = df_pheno

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
    "fpath_imaging",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_IMAGING",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_FL_DATA",
)
@click.option("--decline/--no-decline", "include_decline", default=True)
@click.option("--age/--no-age", "include_age", default=True)
@click.option("--sex/--no-sex", "include_sex", default=False)
@click.option("--diag/--no-diag", "include_diag", default=False)
@click.option("--cases/--no-cases", "include_cases", default=True)
@click.option("--controls/--no-controls", "include_controls", default=False)
@click.option("--aparc/--no-aparc", "include_aparc", default=True)
@click.option("--aseg/--no-aseg", "include_aseg", default=False)
def get_data_qpn(
    fpath_demographics: Path,
    fpath_age: Path,
    fpath_diagnosis: Path,
    fpath_moca: Path,
    fpath_imaging: Path,
    dpath_out: Path,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_diag=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
):
    fname_data_out_components = ["qpn"]
    if include_decline:
        fname_data_out_components.append("decline")
    if include_age:
        fname_data_out_components.append("age")
    if include_sex:
        fname_data_out_components.append("sex")
    if include_diag:
        fname_data_out_components.append("diag")
    if include_cases:
        fname_data_out_components.append("case")
    if include_controls:
        fname_data_out_components.append("hc")
    if include_aparc:
        fname_data_out_components.append("aparc")
    if include_aseg:
        fname_data_out_components.append("aseg")

    dpath_out.mkdir(exist_ok=True)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_qpn = get_df_qpn(
        fpath_demographics=fpath_demographics,
        fpath_age=fpath_age,
        fpath_diagnosis=fpath_diagnosis,
        fpath_moca=fpath_moca,
        fpath_imaging=fpath_imaging,
        include_decline=include_decline,
        include_age=include_age,
        include_sex=include_sex,
        include_diag=include_diag,
        include_cases=include_cases,
        include_controls=include_controls,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
    )
    if df_qpn.empty:
        raise ValueError("Empty dataset")
    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_qpn.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_qpn.shape} to {fpath_data_out}")


if __name__ == "__main__":
    get_data_qpn()
