#!/usr/bin/env python

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TERMURL_PARTICIPANT_ID = "nb:ParticipantID"
TERMURL_AGE = "nb:Age"
TERMURL_MOCA = "enigmapd:trm_038"
TERMURL_MMSE = "enigmapd:trm_039"

# not actually part of ENIGMA-PD vocabulary
TERMURL_COG_DECLINE_STATUS = "fl:cognitive_decline_status"
TERMURL_COG_DECLINE_AVAILABILITY = "fl:cognitive_decline_availability"

# for data dictionary entry
LABEL_COG_DECLINE_STATUS = "Cognitive decline status"
LABEL_COG_DECLINE_AVAILABILITY = "Cognitive decline availability"

COL_COG_DECLINE_STATUS = "cog_decline_status"
COL_COG_DECLINE_AVAILABILITY = "cog_decline_availability"


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Create Neurobagel-harmonized phenotypic data with a cognitive decline column"
    )
    parser.add_argument("tsv_file", type=Path, help="Phenotypic data")
    parser.add_argument("json_file", type=Path, help="Neurobagel data dictionary")
    parser.add_argument(
        "out_file", type=Path, help="Where to write harmonized phenotypic data"
    )
    parser.add_argument(
        "--alg", dest="algorithm_id", type=int, default=1, help="Algorithm ID"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory for intermediate files",
    )
    parser.add_argument(
        "--col-cog-decline",
        type=str,
        help="If the TSV file already has a cognitive decline column (with True/False/NA), specify the column name here",
    )
    return parser


def _get_column_for_termurl(annotations: dict, termurl: str) -> str:
    for col, annots in annotations.items():
        if (
            annots.get("Annotations", {}).get("IsAbout", {}).get("TermURL", None)
            == termurl
        ):
            return col
        elif (
            annots.get("Annotations", {}).get("IsPartOf", {}).get("TermURL", None)
            == termurl
        ):
            return col
    raise ValueError(f"TermURL {termurl} not found in annotations")


def _get_col_score(annotations: dict) -> str:
    col_score = None
    for termurl_score in [TERMURL_MOCA, TERMURL_MMSE]:
        try:
            col_score = _get_column_for_termurl(annotations, termurl_score)
            break
        except ValueError:
            continue
    if col_score is None:
        raise ValueError(f"None of the score TermURLs found in annotations")
    return col_score


def _is_cog_decline(
    df: pd.DataFrame,
    algorithm_id: int,
    col_participant_id: str,
    col_age: str,
    col_score: str,
) -> bool:

    def _is_cog_decline1(df: pd.DataFrame) -> bool:
        col_rate = "rate"  # internal
        if df.index.name != col_participant_id:
            df = df.set_index(col_participant_id)

        df_diff = df.diff().reset_index()
        df = df.reset_index()
        df.index = df_diff.index

        df_diff = df_diff.query(f"{col_age} >= 0.5").copy()
        if df_diff.dropna().empty:
            return pd.NA
        df_diff[col_rate] = df_diff[col_score] / df_diff[col_age]

        # outlier(s): +2/year (improvement)
        if df_diff[col_rate].max() >= 2:
            # find the index of the first outlier diff
            # remove the higher (second) value
            outlier_idx = df_diff.query(
                f"{col_rate} == {df_diff[col_rate].max()}"
            ).index[0]
            df_subset = df.drop(index=outlier_idx)
            return _is_cog_decline1(df_subset)
        else:
            return (df_diff[col_rate] <= -1).any() and (
                min(df[col_score].iloc[1:] - df[col_score].iloc[0]) <= -2
            )

    def _is_cog_decline2(df: pd.DataFrame) -> bool:
        if len(df) < 2 or (df[col_age] - df[col_age].min()).max() < 0.5:
            return pd.NA
        ages = df[col_age]
        scores = df[col_score]
        slope = np.polyfit(ages, scores, 1)[0]
        return slope <= -1

    def _is_cog_decline3(df: pd.DataFrame) -> bool:
        # diff between BL and any follow-up <= -3
        if len(df) < 2:
            return pd.NA
        df_diff = df.iloc[1:] - df.iloc[0]
        if df_diff[col_score].isnull().all():
            return pd.NA
        return df_diff[col_score].min() <= -3

    algorithm_map = {
        1: _is_cog_decline1,
        2: _is_cog_decline2,
        3: _is_cog_decline3,
    }

    df = df.sort_values(by=col_age)

    if df.empty:
        return pd.NA

    df = df.dropna(how="any")
    if df.empty:
        return pd.NA
    df = df.query(f"{col_age} <= {df[col_age].min() + 5}")

    if df.empty:
        return pd.NA

    return algorithm_map[algorithm_id](df)


def _get_df_cog_decline(
    df: pd.DataFrame,
    algorithm_id: int,
    col_participant_id: str,
    col_age: str,
    col_score: str,
) -> pd.DataFrame:
    data_cog_decline = []
    for participant_id in df[col_participant_id].unique():
        df_participant = df.query(
            f"{col_participant_id} == @participant_id and {col_score}.notnull()"
        ).set_index(col_participant_id)[[col_age, col_score]]

        data_cog_decline.append(
            {
                col_participant_id: participant_id,
                COL_COG_DECLINE_STATUS: _is_cog_decline(
                    df_participant,
                    algorithm_id=algorithm_id,
                    col_participant_id=col_participant_id,
                    col_age=col_age,
                    col_score=col_score,
                ),
            }
        )

    return pd.DataFrame(data_cog_decline).set_index(col_participant_id)


def _get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def cog_decline_harmonize(
    tsv_file: Path,
    json_file: Path,
    out_file: Path,
    algorithm_id: int,
    work_dir: Optional[Path] = None,
    col_cog_decline: Optional[str] = None,
):
    if work_dir is None:
        work_dir = Path.cwd()

    annotations = json.loads(json_file.read_text())

    col_participant_id = _get_column_for_termurl(annotations, TERMURL_PARTICIPANT_ID)
    col_age = _get_column_for_termurl(annotations, TERMURL_AGE)
    if col_cog_decline is None:
        col_score = _get_col_score(annotations)
        cols_numeric = (col_age, col_score)
    else:
        cols_numeric = (col_age,)

    df = pd.read_csv(
        tsv_file,
        sep="\t",
        dtype={col: str for col in annotations.keys() if col not in cols_numeric},
    )

    if col_cog_decline is None:
        df_cog_decline = _get_df_cog_decline(
            df,
            algorithm_id=algorithm_id,
            col_participant_id=col_participant_id,
            col_age=col_age,
            col_score=col_score,
        )

        # add cognitive decline column to df
        df = df.set_index(col_participant_id)
        df = df.merge(
            df_cog_decline,
            left_index=True,
            right_index=True,
            how="left",
        )
        col_cog_decline = COL_COG_DECLINE_STATUS
    df[COL_COG_DECLINE_AVAILABILITY] = ~df[col_cog_decline].isna()

    # # remove "sub-" prefix if present
    # df[col_participant_id] = df[col_participant_id].apply(
    #     lambda x: str(x).removeprefix("sub-")
    # )

    # add cognitive decline column annotation to json
    annotations[col_cog_decline] = {
        "Description": f"Whether the participant shows cognitive decline based on Algorithm {algorithm_id}.",
        "Annotations": {
            "IsAbout": {"TermURL": "nb:Assessment", "Label": "Assessment Tool"},
            "VariableType": "Collection",
            "IsPartOf": {
                "TermURL": TERMURL_COG_DECLINE_STATUS,
                "Label": LABEL_COG_DECLINE_STATUS,
            },
            "MissingValues": ["", "False"],
        },
    }
    annotations[COL_COG_DECLINE_AVAILABILITY] = {
        "Description": "Whether cognitive decline status is defined.",
        "Annotations": {
            "IsAbout": {"TermURL": "nb:Assessment", "Label": "Assessment Tool"},
            "VariableType": "Collection",
            "IsPartOf": {
                "TermURL": TERMURL_COG_DECLINE_AVAILABILITY,
                "Label": LABEL_COG_DECLINE_AVAILABILITY,
            },
            "MissingValues": ["False"],
        },
    }

    # write intermediate files
    timestamp = _get_timestamp()
    fpath_out_tsv = work_dir / f"pheno_with_cognitive_decline-{timestamp}.tsv"
    fpath_out_json = work_dir / f"pheno_with_cognitive_decline-{timestamp}.json"
    work_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(fpath_out_tsv, sep="\t", index=True)
    fpath_out_json.write_text(json.dumps(annotations, indent=4))

    # call bagel CLI
    subprocess.run(
        [
            "bagel",
            "harmonize-pheno",
            "--pheno",
            str(fpath_out_tsv),
            "--dictionary",
            str(fpath_out_json),
            "--output",
            str(out_file),
            "--overwrite",
        ],
        check=True,
    )

    print(f"Saved harmonized phenotypic data to {out_file}")


if __name__ == "__main__":
    args = _get_parser().parse_args()
    cog_decline_harmonize(
        tsv_file=args.tsv_file,
        json_file=args.json_file,
        out_file=args.out_file,
        algorithm_id=args.algorithm_id,
        work_dir=args.work_dir,
        col_cog_decline=args.col_cog_decline,
    )
