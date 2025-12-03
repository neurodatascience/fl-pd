#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_pd.utils.constants import CLICK_CONTEXT_SETTINGS


COL_AGE = "age"
COL_VISIT_ID = "visit_id"
COL_MOCA = "moca"


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_demographics",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_DEMOGRAPHICS",
)
@click.argument(
    "fpath_mri",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_QPN_MRI",
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
    "fpath_out",
    type=click.Path(path_type=Path, writable=True),
    envvar="FPATH_QPN_PHENO_CLEAN",
)
def get_data_qpn(
    fpath_demographics: Path,
    fpath_mri: Path,
    fpath_diagnosis: Path,
    fpath_moca: Path,
    fpath_out: Path,
):
    # static
    df_demographics = pd.read_csv(fpath_demographics).set_index("participant_id")[
        ["enrollment_group", "sex", "education_years", "education_level"]
    ]
    df_diagnosis = pd.read_csv(fpath_diagnosis).set_index("participant_id")[
        ["diagnosis_group_for_analysis", "diagnosis_age"]
    ]

    # non-static: MRI age
    df_mri = pd.read_csv(fpath_mri).set_index("participant_id")
    df_mri[COL_VISIT_ID] = df_mri["session"].apply(
        lambda x: str(x).removeprefix("ses-")
    )
    df_mri = df_mri.rename(columns={"MRI_age": COL_AGE})
    df_mri = df_mri[[COL_VISIT_ID, COL_AGE]]
    df_mri = df_mri.set_index(COL_VISIT_ID, append=True)

    # non-static: MoCA score and MoCA age
    df_moca = pd.read_csv(fpath_moca).set_index("participant_id")
    df_moca = df_moca.rename(
        columns={
            "redcap_event_name": COL_VISIT_ID,
            "MoCA Total Score": COL_MOCA,
            "moca_age": COL_AGE,
        }
    )
    df_moca = df_moca[[COL_VISIT_ID, COL_MOCA, COL_AGE]]
    df_moca = df_moca.set_index(COL_VISIT_ID, append=True)

    df_pheno_static = df_demographics.merge(
        df_diagnosis, left_index=True, right_index=True, how="outer"
    )

    df_pheno_nonstatic = pd.concat([df_mri, df_moca], axis="index")

    df_pheno = df_pheno_static.merge(
        df_pheno_nonstatic.reset_index(level=COL_VISIT_ID),
        left_index=True,
        right_index=True,
        how="outer",
    )

    df_pheno = df_pheno.set_index(COL_VISIT_ID, append=True)
    df_pheno = df_pheno.sort_index()

    fpath_out.parent.mkdir(exist_ok=True, parents=True)
    df_pheno.to_csv(fpath_out, index=True, header=True, sep="\t")
    print(f"Saved final dataframe of shape {df_pheno.shape} to {fpath_out}")


if __name__ == "__main__":
    get_data_qpn()
