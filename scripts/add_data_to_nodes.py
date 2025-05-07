#!/usr/bin/env python
from pathlib import Path
from typing import Tuple
import configparser
import re
import json
import subprocess
import tempfile
import warnings

import click
import pandas as pd

from fl_pd.utils.constants import DATASETS, ML_PROBLEM_MAP

NODE_MAP = {
    "mega": "node-mega",
    "adni": "node-adni",
    "ppmi": "node-ppmi",
    "qpn": "node-qpn",
}


def _get_dataset_name(fname) -> str:
    for dataset_name in ("mega",) + DATASETS:
        if dataset_name in fname:
            return dataset_name
    raise ValueError(f"No dataset name found for {fname=}")


def _get_data_tag(fname) -> str:
    for substring in ML_PROBLEM_MAP.keys():
        if substring in fname:
            return substring
    raise ValueError(f"Invalid data tags for {fname=}")


def _get_i_train(fname) -> int:
    if (match := re.search("(\d+)train", fname)) is not None:
        groups = match.groups()
        if len(groups) == 1:
            return int(groups[0])
    raise ValueError(f"No i_train found for {fname=}")


def _get_data_info(fname) -> Tuple[str, str, int]:
    return (_get_dataset_name(fname), _get_data_tag(fname), _get_i_train(fname))


def _get_tags(dataset_name: str, data_tag: str, i_train: int) -> str:
    tags = []
    if dataset_name != "mega":
        tags.append("federated")
    tags.extend([dataset_name, data_tag, f"{i_train}train"])
    return ",".join(tags)


def _data_already_added(dpath_node: Path, fpath_tsv: Path) -> bool:
    fpath_config = dpath_node / "etc" / "config.ini"
    if not fpath_config.exists():
        raise FileNotFoundError(f"Node config file not found: {fpath_config}")

    config = configparser.ConfigParser()
    config.read(str(fpath_config))
    fpath_db = (fpath_config.parent / config["default"]["db"]).resolve()
    if not fpath_db.exists():
        warnings.warn(f"Node database file not found: {fpath_db}")
        return False

    db_json = json.loads(fpath_db.read_text())
    df_datasets = pd.DataFrame(db_json["Datasets"]).T

    return str(fpath_tsv) in df_datasets["path"].to_list()


def _add_data_to_node(fpath_tsv: Path, dpath_nodes: Path):
    dataset_name, data_tag, i_train = _get_data_info(fpath_tsv.name)
    dpath_node = dpath_nodes / f"node-{dataset_name}"

    if _data_already_added(dpath_node, fpath_tsv):
        print(f"{fpath_tsv.name} is already in node {dpath_node.name}. Skipping")
        return

    dataset_info = {
        "path": str(fpath_tsv),
        "data_type": "csv",
        "description": "",
        "tags": _get_tags(dataset_name, data_tag, i_train),
        "name": f"{dataset_name.upper()} {data_tag} (train {i_train})",
    }
    with tempfile.NamedTemporaryFile(mode="+wt") as file_json:
        file_json.write(json.dumps(dataset_info))
        file_json.flush()
        subprocess.run(
            [
                "fedbiomed",
                "node",
                "-p",
                str(dpath_node),
                "dataset",
                "add",
                "--file",
                file_json.name,
            ],
            check=True,
        )


@click.command()
@click.argument(
    "dpath_data",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FL_DATA",
)
@click.argument(
    "dpath_nodes",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    envvar="DPATH_FEDBIOMED",
)
def add_data_to_nodes(dpath_data: Path, dpath_nodes: Path):
    fpaths_tsv = dpath_data.glob("**/*train*.tsv")
    for fpath_tsv in sorted(fpaths_tsv):
        print(f"----- {fpath_tsv} -----")
        _add_data_to_node(fpath_tsv, dpath_nodes)


if __name__ == "__main__":
    add_data_to_nodes()
