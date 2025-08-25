#!/usr/bin/env python

from pathlib import Path
from typing import Iterable, Optional

import click
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS, COLS_PHENO


def _get_aseg_cols(cols: Iterable[str]):
    cols_not_aseg = COLS_PHENO + _get_aparc_cols(cols)
    return [col for col in cols if col not in cols_not_aseg]


def _get_aparc_cols(cols: Iterable[str]):
    return [col for col in cols if "thickness" in col]


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument(
    "data_tags",
    type=str,
    nargs=-1,
)
@click.option("--decline/--no-decline", "include_decline", default=True)
@click.option("--age/--no-age", "include_age", default=True)
@click.option("--sex/--no-sex", "include_sex", default=False)
@click.option("--diag/--no-diag", "include_diag", default=False)
@click.option("--aseg/--no-aseg", "include_aseg", default=False)
@click.option("--aparc/--no-aparc", "include_aparc", default=True)
@click.option("--cases/--no-cases", "include_cases", default=True)
@click.option("--controls/--no-controls", "include_controls", default=False)
@click.option("--dropna", "col_target", type=str, multiple=True, default=None)
def subset_data(
    dpath_data: Path,
    data_tags: Iterable[str],
    include_decline=True,
    include_age=True,
    include_sex=False,
    include_diag=False,
    include_aseg=False,
    include_aparc=True,
    include_cases=True,
    include_controls=False,
    col_target: Optional[str] = None,
):
    if len(data_tags) == 0:
        raise click.UsageError(
            "No data tags provided. Please specify at least one tag."
        )

    fname_data_out_components = []
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

    for data_tag in data_tags:
        print(f"===== {data_tag} =====")

        fpath_data = dpath_data / data_tag / f"{data_tag}.tsv"
        if not fpath_data.exists():
            raise click.UsageError(f"Data file {fpath_data} does not exist.")

        df_full = pd.read_csv(fpath_data, sep="\t", index_col="participant_id")
        print(f"Full dataframe shape: {df_full.shape}")

        cols = []
        if include_decline:
            cols.append("COG_DECLINE")
        if include_age:
            cols.append("AGE")
        if include_sex:
            cols.append("SEX")
        if include_diag:
            cols.append("DIAGNOSIS")
        if include_aseg:
            cols.extend(_get_aseg_cols(df_full.columns))
        if include_aparc:
            cols.extend(_get_aparc_cols(df_full.columns))

        if col_target is not None:
            df_full = df_full.dropna(subset=col_target, axis="index")
            print(f"Shape after dropping NAs in {col_target}: {df_full.shape}")

        dfs_to_concatenate = []
        if include_controls:
            dfs_to_concatenate.append(df_full.query("IS_CONTROL == True"))
            print(f"Keeping {len(dfs_to_concatenate[-1])} controls")
        if include_cases:
            dfs_to_concatenate.append(df_full.query("IS_CASE == True"))
            print(f"Keeping {len(dfs_to_concatenate[-1])} cases")
        df_subset = pd.concat(dfs_to_concatenate, axis="index")
        df_subset = df_subset.drop_duplicates()

        print(f"Keeping {len(cols)} columns")
        df_subset = df_subset.loc[:, cols]

        print(f"Final dataframe shape: {df_subset.shape}")

        subset_tag = "-".join([data_tag] + fname_data_out_components)
        fpath_data_out = (dpath_data / subset_tag / subset_tag).with_suffix(".tsv")
        fpath_data_out.parent.mkdir(parents=True, exist_ok=True)
        df_subset.to_csv(fpath_data_out, sep="\t", index=True, header=True)


if __name__ == "__main__":
    subset_data()
