#!/usr/bin/env python

from pathlib import Path

import click
import numpy as np
import pandas as pd

from utils import (
    CLICK_CONTEXT_SETTINGS,
    fs6_to_fs7,
    fs7_aparc_to_keep,
    fs7_aseg_to_keep,
)


def get_df_pheno(
    fpath_pheno,
    include_decline=True,
    include_age=True,
    include_sex=False,
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

    df_pheno = pd.read_csv(fpath_pheno, low_memory=False)
    df_pheno["participant_id"] = "ADNI" + df_pheno["PTID"].str.replace("_", "")
    df_pheno = df_pheno.set_index("participant_id")

    df_pheno["EXAMDATE"] = pd.to_datetime(df_pheno["EXAMDATE"])
    df_pheno = df_pheno.sort_values(by="EXAMDATE", ascending=True)

    # use PPMI coding for sex
    df_pheno["SEX"] = (
        df_pheno["PTGENDER"].map({"Male": 1, "Female": 0}).astype("category")
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

    if not include_controls:
        df_pheno = df_pheno.query("DX_bl != 'CN'")
    if not include_cases:
        df_pheno = df_pheno.query("DX_bl == 'CN'")

    df_pheno = df_pheno[cols]
    print(f"Keeping phenotypic columns: {cols}")

    return df_pheno


def get_df_imaging(
    fpath_imaging,
    include_aparc=True,
    include_aseg=False,
):
    df_imaging = pd.read_csv(fpath_imaging, sep="\t", dtype={"participant_id": str})
    df_imaging = df_imaging.set_index(["participant_id", "session_id"])
    df_imaging = fs6_to_fs7(df_imaging)

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
    df_imaging = df_imaging.query('session_id == "bl"').reset_index(
        "session_id", drop=True
    )

    return df_imaging


def get_df_adni(
    fpath_pheno,
    fpath_imaging,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
):
    df_pheno = get_df_pheno(
        fpath_pheno,
        include_decline=include_decline,
        include_age=include_age,
        include_sex=include_sex,
        include_cases=include_cases,
        include_controls=include_controls,
    )
    df_imaging = get_df_imaging(
        fpath_imaging,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
    )
    if include_aparc or include_aseg:
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
    "fpath_imaging",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_ADNI_IMAGING",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="DPATH_FL_DATA",
)
@click.option("--decline/--no-decline", "include_decline", default=True)
@click.option("--age/--no-age", "include_age", default=True)
@click.option("--sex/--no-sex", "include_sex", default=False)
@click.option("--cases/--no-cases", "include_cases", default=True)
@click.option("--controls/--no-controls", "include_controls", default=False)
@click.option("--aparc/--no-aparc", "include_aparc", default=True)
@click.option("--aseg/--no-aseg", "include_aseg", default=False)
def get_data_adni(
    fpath_pheno,
    fpath_imaging,
    dpath_out,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
):

    fname_data_out_components = ["adni"]
    if include_decline:
        fname_data_out_components.append("decline")
    if include_age:
        fname_data_out_components.append("age")
    if include_sex:
        fname_data_out_components.append("sex")
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

    df_adni = get_df_adni(
        fpath_pheno=fpath_pheno,
        fpath_imaging=fpath_imaging,
        include_decline=include_decline,
        include_age=include_age,
        include_sex=include_sex,
        include_cases=include_cases,
        include_controls=include_controls,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
    )
    if df_adni.empty:
        raise ValueError("Empty dataset")

    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_adni.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_adni.shape} to {fpath_data_out}")


if __name__ == "__main__":

    get_data_adni()
