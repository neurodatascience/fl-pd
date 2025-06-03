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

VISIT_IDS_ORDERED = ["SC", "BL", "V04", "V06", "V08", "V10"]


def get_df_pheno(fpath_pheno):

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

    # convert to datetime
    for col in ["DATE_OF_BIRTH", "DATE_OF_DIAGNOSIS"]:
        df_pheno[col] = pd.to_datetime(df_pheno[col], format="%m/%Y", errors="coerce")

    # convert to categorical
    for col in ["SEX", "COHORT_DEFINITION", "PRIMARY_DIAGNOSIS"]:
        df_pheno[col] = df_pheno[col].astype("category")

    # determine cognitive decline
    # criterion: drop of >4 from first visit to any of the follow-ups
    data_for_df_moca_diff = []
    for participant_id in df_pheno.index.get_level_values("participant_id").unique():
        df_participant = df_pheno.loc[participant_id]
        moca_values = df_participant["MOCA"].dropna()
        if len(moca_values) < 2:
            continue

        moca_diffs = moca_values.iloc[1:] - moca_values.iloc[0]

        data_for_df_moca_diff.append(
            {
                "participant_id": participant_id,
                "MOCA_DIFF": moca_diffs.min(),
            }
        )

    df_moca_diff = pd.DataFrame(data_for_df_moca_diff)
    df_pheno = (
        df_pheno.reset_index("visit_id")
        .merge(
            df_moca_diff.set_index("participant_id"),
            left_index=True,
            right_index=True,
            how="left",
        )
        .set_index("visit_id", append=True)
    )
    df_pheno["COG_DECLINE"] = df_pheno["MOCA_DIFF"].apply(
        lambda x: x <= -3 if not np.isnan(x) else np.nan
    )

    # fill in missing BL values with SC (if available) for the MOCA
    for idx, row in df_pheno.query('visit_id == "BL"').iterrows():
        col = "MOCA"
        if pd.isna(row[col]):
            try:
                sc_row = df_pheno.loc[(idx[0], "SC")]
            except KeyError:
                continue

            sc_val = sc_row[col].item()
            df_pheno.loc[idx, col] = sc_val

    return df_pheno


def get_df_imaging(
    fpath_aseg,
    fpath_aparc,
    include_aparc=True,
    include_aseg=False,
    is_fs6=False,
):

    df_aseg = pd.read_csv(fpath_aseg, sep="\t", dtype={"participant_id": str})
    df_aseg = df_aseg.set_index(["participant_id", "session_id"])

    df_aparc = pd.read_csv(fpath_aparc, sep="\t", dtype={"participant_id": str})
    df_aparc = df_aparc.set_index(["participant_id", "session_id"])

    if is_fs6:
        df_aseg = fs6_to_fs7(df_aseg)
        df_aparc = fs6_to_fs7(df_aparc)

    df_aseg = fs7_aseg_to_keep(df_aseg)
    df_aparc = fs7_aparc_to_keep(df_aparc)

    dfs_imaging = []
    if include_aseg:
        dfs_imaging.append(df_aseg)
    if include_aparc:
        dfs_imaging.append(df_aparc)

    return pd.concat(dfs_imaging, axis="columns")


def get_df_ppmi(
    fpath_pheno,
    fpath_aseg,
    fpath_aparc,
    dpath_normative_modelling_data,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_cases=True,
    include_controls=False,
    include_diag=False,
    include_aparc=True,
    include_aseg=False,
    is_fs6=False,
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
    df_pheno = df_pheno_full.copy()

    # keep PD patients + prodromal
    if not include_controls:
        df_pheno = df_pheno.query(
            'COHORT_DEFINITION == "Parkinson\'s Disease" or COHORT_DEFINITION == "Prodromal"'
        )
    if not include_cases:
        df_pheno = df_pheno.query('COHORT_DEFINITION == "Healthy Control"')
    print(f"Keeping phenotypic columns: {cols}")
    df_pheno = df_pheno[cols]

    df_imaging = get_df_imaging(
        fpath_aseg,
        fpath_aparc,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
        is_fs6=is_fs6,
    )
    df_imaging.index.names = df_pheno.index.names

    # merge
    df_pheno.index = df_pheno.index.rename(df_imaging.index.names)
    if include_aparc or include_aseg:
        if apply_normative_modelling:
            df_merged_full = df_pheno_full.loc[:, ["AGE", "SEX", "DIAGNOSIS"]].merge(
                df_imaging, left_index=True, right_index=True, how="right"
            )
            df_imaging = get_z_scores(
                df_merged_full.drop(columns=["DIAGNOSIS"]),
                df_merged_full.query("DIAGNOSIS == 0").drop(columns=["DIAGNOSIS"]),
                dpath_normative_modelling_data,
            )
        df_merged = df_pheno.merge(
            df_imaging, left_index=True, right_index=True, how="inner"
        )

    else:
        df_merged = df_pheno

    # keep only BL
    df_merged = df_merged.query('visit_id == "BL"').reset_index("visit_id", drop=True)

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
@click.option("--norm/--no-norm", "apply_normative_modelling", default=False)
def get_data_ppmi(
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
    fpath_aseg_fs6=None,
    fpath_aparc_fs6=None,
    use_fs6=False,
    apply_normative_modelling=False,
):

    if use_fs6 and fpath_aseg_fs6 is None:
        raise ValueError("Must provide path to FS6 imaging data if --fs6 is given")

    fname_data_out_components = ["ppmi"]
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

    if use_fs6:
        fpath_aseg = fpath_aseg_fs6
        fpath_aparc = fpath_aparc_fs6
        fname_data_out_components.append("fs6")

    dpath_out.mkdir(exist_ok=True)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_ppmi = get_df_ppmi(
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
        is_fs6=use_fs6,
        apply_normative_modelling=apply_normative_modelling,
    )
    if df_ppmi.empty:
        raise ValueError("Empty dataset")
    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_ppmi.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_ppmi.shape} to {fpath_data_out}")


if __name__ == "__main__":
    get_data_ppmi()
