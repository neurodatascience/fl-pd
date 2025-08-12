#!/usr/bin/env python

from pathlib import Path
from typing import Iterable

import click
import numpy as np
import pandas as pd

from fl_pd.io import get_dpath_latest

DEFAULT_N_SITES = 3
DEFAULT_DNAME = "_fedbiomed_simulated"
DEFAULT_FNAMES = [
    "adni_client1.csv",
    "adni_client2.csv",
    "adni_client3.csv",
    "adni_validation.csv",
]


@click.command()
@click.argument(
    "dpath_data",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    envvar="DPATH_FL_DATA",
)
@click.option("--n-sites", type=int, default=DEFAULT_N_SITES)
@click.option("--dirname", "dname", default=DEFAULT_DNAME)
@click.option("--filename", "fnames", multiple=True, default=DEFAULT_FNAMES)
def get_data_simulated(
    dpath_data: Path, n_sites: int, dname: str, fnames: Iterable[Path]
):
    dfs = []
    for fname in fnames:
        fpath = dpath_data / dname / fname
        print(f"Loading dataframe from {fpath}")
        dfs.append(pd.read_csv(fpath, index_col=0))

    print("-" * 20)
    df_all = pd.concat(dfs)
    df_all = df_all.rename(columns={"MMSE.bl": "MMSE"})
    df_all.index.name = "participant_id"
    print(f"{df_all.shape=}")
    print("-" * 20)

    dfs_sites: list[pd.DataFrame] = np.array_split(df_all, n_sites)

    dpath_data = get_dpath_latest(dpath_data)
    for i_site, df_site in enumerate(dfs_sites, 1):
        print(f"Site {i_site}: {df_site.shape}")
        tag = f"site{i_site}-simulated"
        fpath_out = dpath_data / tag / f"{tag}.tsv"
        fpath_out.parent.mkdir(exist_ok=True)
        df_site.to_csv(fpath_out, index=True, header=True, sep="\t")
        print(f"Saved to {fpath_out}")


if __name__ == "__main__":
    get_data_simulated()
