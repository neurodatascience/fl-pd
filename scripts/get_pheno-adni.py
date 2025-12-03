#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_pheno",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_ADNI_PHENO",
)
@click.argument(
    "fpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="FPATH_ADNI_PHENO_CLEAN",
)
def get_pheno_adni(fpath_pheno: Path, fpath_out: Path):

    df_pheno = pd.read_csv(fpath_pheno, low_memory=False)
    df_pheno.insert(0, "participant_id", "ADNI" + df_pheno["PTID"].str.replace("_", ""))
    df_pheno["AGE"] = df_pheno["AGE"] + df_pheno["Years_bl"]

    fpath_out.parent.mkdir(exist_ok=True, parents=True)
    df_pheno.to_csv(fpath_out, sep="\t", index=False)
    print(f"Saved final dataframe of shape {df_pheno.shape} to {fpath_out}")


if __name__ == "__main__":

    get_pheno_adni()
