#!/usr/bin/env python

from pathlib import Path

import click
import pandas as pd

from fl_pd.io import get_dpath_latest
from fl_pd.pheno import cog_decline_from_pad_mci
from fl_pd.utils.constants import (
    CLICK_CONTEXT_SETTINGS,
    DPATH_RELATIVE_PAD_IMAGING_SESSIONS,
)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.argument(
    "fpath_manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_MANIFEST",
)
@click.argument(
    "fpath_mci",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    envvar="FPATH_PAD_MCI",
)
@click.argument(
    "dpath_out",
    type=click.Path(path_type=Path, dir_okay=True),
    envvar="DPATH_FL_DATA",
)
def choose_pad_imaging_sessions(fpath_manifest: Path, fpath_mci: Path, dpath_out: Path):
    df_manifest = pd.read_csv(fpath_manifest, sep="\t")
    df_mci = pd.read_csv(fpath_mci, sep="\t")

    # ignore enrollment sessions
    df_manifest["n_months"] = df_manifest["visit_id"].str.extract(r"(\d+)").astype(int)
    df_manifest = df_manifest.query('visit_id != "NAPEN00" and visit_id != "PREEN00"')

    # index is participant_id
    df_cog_decline = cog_decline_from_pad_mci(df_mci)

    data_imaging_sessions = []
    for participant_id, df_manifest_participant in df_manifest.groupby(
        "participant_id"
    ):
        df_manifest_participant = df_manifest_participant.sort_values("n_months")

        if participant_id not in df_cog_decline.index:
            print(f"Warning: No MCI info for participant {participant_id}")
            session_id = df_manifest_participant["visit_id"].iloc[0]

        elif df_cog_decline.loc[participant_id, "COG_DECLINE"]:
            n_months_switch = df_cog_decline.loc[
                participant_id, "n_months_switch"
            ].item()
            df_manifest_participant = df_manifest_participant.query(
                "n_months < @n_months_switch"
            )
            session_id = df_manifest_participant["visit_id"].iloc[-1]
        else:
            session_id = df_manifest_participant["visit_id"].iloc[0]

        data_imaging_sessions.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
            }
        )

    df_imaging_sessions = pd.DataFrame(data_imaging_sessions)
    print(df_imaging_sessions.reset_index()["session_id"].value_counts(dropna=False))

    fpath_out = get_dpath_latest(dpath_out) / DPATH_RELATIVE_PAD_IMAGING_SESSIONS

    fpath_out.parent.mkdir(parents=True, exist_ok=True)
    df_imaging_sessions.to_csv(fpath_out, header=None, index=None, sep="\t")
    print(f"Saved participant-session list {df_imaging_sessions.shape} to {fpath_out}")


if __name__ == "__main__":
    choose_pad_imaging_sessions()
