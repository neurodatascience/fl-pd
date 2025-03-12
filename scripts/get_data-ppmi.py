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

VISIT_IDS_ORDERED = ["SC", "BL", "V04", "V06", "V08", "V10"]


def get_df_pheno(
    fpath_pheno,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_cases=True,
    include_controls=False,
):
    # other cols: "EDUCATION", "MOCA"
    cols = []
    if include_decline:
        cols.append("COG_DECLINE")
    if include_age:
        cols.append("AGE")
    if include_sex:
        cols.append("SEX")

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

    # convert to datetime
    for col in ["DATE_OF_BIRTH", "DATE_OF_DIAGNOSIS"]:
        df_pheno[col] = pd.to_datetime(df_pheno[col], format="%m/%Y", errors="coerce")

    # convert to categorical
    for col in ["SEX", "COHORT_DEFINITION", "PRIMARY_DIAGNOSIS"]:
        df_pheno[col] = df_pheno[col].astype("category")

    # keep PD patients + prodromal
    if not include_controls:
        df_pheno = df_pheno.query(
            'COHORT_DEFINITION == "Parkinson\'s Disease" or COHORT_DEFINITION == "Prodromal"'
        )
    if not include_cases:
        df_pheno = df_pheno.query('COHORT_DEFINITION == "Healthy Control"')

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

    print(f"Keeping phenotypic columns: {cols}")
    return df_pheno[cols]


def get_df_imaging(
    fpath_imaging,
    include_aparc=True,
    include_aseg=False,
    is_fs6=False,
):

    df_imaging = pd.read_csv(fpath_imaging, sep="\t", dtype={"participant_id": str})
    df_imaging = df_imaging.set_index(["participant_id", "session_id"])

    if is_fs6:
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

    return pd.concat(dfs_imaging, axis="columns")


def get_df_ppmi(
    fpath_pheno,
    fpath_imaging,
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_cases=True,
    include_controls=False,
    include_aparc=True,
    include_aseg=False,
    is_fs6=False,
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
        is_fs6=is_fs6,
    )

    # merge
    df_pheno.index = df_pheno.index.rename(df_imaging.index.names)
    if include_aparc or include_aseg:
        df_merged = df_pheno.merge(
            df_imaging, left_index=True, right_index=True, how="inner"
        )
    else:
        df_merged = df_pheno

    # keep only BL
    df_merged = df_merged.query('session_id == "BL"').reset_index(
        "session_id", drop=True
    )

    return df_merged


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_pheno",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PPMI_PHENO",
)
@click.argument(
    "fpath_imaging",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PPMI_IMAGING",
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
@click.option("--fs6/--fs7", "use_fs6", default=False)
@click.option(
    "--fpath-imaging-fs6",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=False,
    envvar="FPATH_PPMI_IMAGING_FS6",
)
def get_data_ppmi(
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
    fpath_imaging_fs6=None,
    use_fs6=False,
):

    if use_fs6 and fpath_imaging_fs6 is None:
        raise ValueError("Must provide path to FS6 imaging data if --fs6 is given")

    fname_data_out_components = ["ppmi"]
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

    if use_fs6:
        fpath_imaging = fpath_imaging_fs6
        fname_data_out_components.append("fs6")

    dpath_out.mkdir(exist_ok=True)
    tags = "-".join(fname_data_out_components)
    fpath_data_out = (dpath_out / tags / tags).with_suffix(".tsv")

    df_ppmi = get_df_ppmi(
        fpath_pheno,
        fpath_imaging,
        include_decline=include_decline,
        include_age=include_age,
        include_sex=include_sex,
        include_cases=include_cases,
        include_controls=include_controls,
        include_aparc=include_aparc,
        include_aseg=include_aseg,
        is_fs6=use_fs6,
    )
    if df_ppmi.empty:
        raise ValueError("Empty dataset")
    fpath_data_out.parent.mkdir(exist_ok=True, parents=True)
    df_ppmi.to_csv(fpath_data_out, sep="\t")
    print(f"Saved final dataframe of shape {df_ppmi.shape} to {fpath_data_out}")


if __name__ == "__main__":
    get_data_ppmi()
