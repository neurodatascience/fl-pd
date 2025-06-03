#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from fl_pd.normative_modelling import get_z_scores
from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_pd.utils.freesurfer import (
    fs6_to_fs7,
    fs7_aparc_to_keep,
    fs7_aseg_to_keep,
)


def get_df_pheno(fpath_pheno):

    df_pheno = pd.read_csv(fpath_pheno, low_memory=False)
    df_pheno["participant_id"] = "ADNI" + df_pheno["PTID"].str.replace("_", "")
    df_pheno = df_pheno.set_index("participant_id")

    df_pheno["EXAMDATE"] = pd.to_datetime(df_pheno["EXAMDATE"])
    df_pheno = df_pheno.sort_values(by="EXAMDATE", ascending=True)

    # use PPMI coding for sex
    df_pheno["SEX"] = (
        df_pheno["PTGENDER"].map({"Male": 1, "Female": 0}).astype("category")
    )

    # recode diagnosis
    df_pheno["DIAGNOSIS"] = (
        df_pheno["DX_bl"]
        .map(lambda x: 0 if x == "CN" else (1 if not pd.isna(x) else pd.NA))
        .astype("category")
    )

    visits = [
        "bl",
        "m0",
        "m03",
        "m06",
        "m12",
        "m18",
        "m24",
        "m30",
        "m36",
        "m42",
        "m48",
        "m54",
        "m60",
    ]
    df_pheno_5_years = df_pheno.query("VISCODE in @visits")
    data_for_df_mmse_diff = []
    for participant_id in df_pheno_5_years.index.get_level_values(
        "participant_id"
    ).unique():
        df_participant = df_pheno_5_years.loc[[participant_id]]

        mmse_values = df_participant["MMSE"].dropna()
        if len(mmse_values) < 2:
            continue

        mmse_diffs = mmse_values.iloc[1:] - mmse_values.iloc[0]

        data_for_df_mmse_diff.append(
            {
                "participant_id": participant_id,
                "MMSE_DIFF": mmse_diffs.min(),
                "n_mmse": len(mmse_values),
            }
        )

    df_mmse_diff = pd.DataFrame(data_for_df_mmse_diff)
    df_pheno = df_pheno.merge(
        df_mmse_diff.set_index("participant_id"),
        left_index=True,
        right_index=True,
        how="left",
    )
    df_pheno["COG_DECLINE"] = df_pheno["MMSE_DIFF"].apply(
        lambda x: x <= -3 if not np.isnan(x) else np.nan
    )

    df_pheno = df_pheno.query('VISCODE == "bl"')

    return df_pheno


def get_df_imaging(
    fpath_aseg,
    fpath_aparc,
    include_aparc=True,
    include_aseg=False,
):
    df_aparc = pd.read_csv(fpath_aparc, sep="\t", dtype={"participant_id": str})
    df_aparc = df_aparc.set_index(["participant_id", "session_id"])
    df_aparc = fs6_to_fs7(df_aparc)
    df_aparc = fs7_aparc_to_keep(df_aparc)

    df_aseg = pd.read_csv(fpath_aseg, sep="\t", dtype={"participant_id": str})
    df_aseg = df_aseg.set_index(["participant_id", "session_id"])
    df_aseg = fs6_to_fs7(df_aseg)
    df_aseg = fs7_aseg_to_keep(df_aseg)

    dfs_imaging = []
    if include_aparc:
        dfs_imaging.append(df_aparc)
    if include_aseg:
        dfs_imaging.append(df_aseg)

    df_aparc = pd.concat(dfs_imaging, axis="columns")
    df_aparc = df_aparc.query('session_id == "bl"').reset_index("session_id", drop=True)

    return df_aparc


def get_df_adni(
    fpath_pheno,
    fpath_aseg,
    fpath_aparc,
    dpath_normative_modelling_data: Path,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_diag=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
    apply_normative_modelling=False,
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

    df_pheno_full = get_df_pheno(fpath_pheno)

    df_pheno = df_pheno_full
    if not include_controls:
        df_pheno = df_pheno.query("DIAGNOSIS == 0")
    if not include_cases:
        df_pheno = df_pheno.query("DIAGNOSIS == 1")
    df_pheno = df_pheno_full.loc[:, cols]
    print(f"Keeping phenotypic columns: {cols}")

    df_imaging = get_df_imaging(
        fpath_aseg,
        fpath_aparc,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
    )
    if include_aparc or include_aseg:
        if apply_normative_modelling:
            df_merged_full = df_pheno_full.loc[:, ["AGE", "SEX", "DIAGNOSIS"]].join(
                df_imaging, how="inner"
            )
            df_imaging = get_z_scores(
                df_merged_full.drop(columns=["DIAGNOSIS"]),
                df_merged_full.query("DIAGNOSIS == 0").drop(columns=["DIAGNOSIS"]),
                dpath_normative_modelling_data,
            )
        df_merged = df_pheno.join(df_imaging, how="inner")
    else:
        df_merged = df_pheno

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
    "dpath_normative_modelling_data",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_NORMATIVE_MODELLING_DATA",
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
@click.option("--norm/--no-norm", "apply_normative_modelling", default=False)
def get_data_adni(
    fpath_pheno: Path,
    fpath_aseg: Path,
    fpath_aparc: Path,
    dpath_normative_modelling_data: Path,
    dpath_out: Path,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_diag=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
    apply_normative_modelling=False,
):

    fname_data_out_components = ["adni"]
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
        if apply_normative_modelling:
            fname_data_out_components.append("aparc_norm")
        else:
            fname_data_out_components.append("aparc")
    if include_aseg:
        if apply_normative_modelling:
            fname_data_out_components.append("aseg_norm")
        else:
            fname_data_out_components.append("aseg")
    dpath_out.mkdir(exist_ok=True)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_adni = get_df_adni(
        fpath_pheno=fpath_pheno,
        fpath_aseg=fpath_aseg,
        fpath_aparc=fpath_aparc,
        dpath_normative_modelling_data=dpath_normative_modelling_data,
        include_decline=include_decline,
        include_age=include_age,
        include_sex=include_sex,
        include_diag=include_diag,
        include_cases=include_cases,
        include_controls=include_controls,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
        apply_normative_modelling=apply_normative_modelling,
    )
    if df_adni.empty:
        raise ValueError("Empty dataset")

    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_adni.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_adni.shape} to {fpath_data_out}")


if __name__ == "__main__":

    get_data_adni()
