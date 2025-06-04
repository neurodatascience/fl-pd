import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pcntoolkit.normative import predict
    from pcntoolkit.util.utils import create_design_matrix
except ImportError:
    raise ImportError(
        "The pcntoolkit package is not installed in this environment. "
        "It has dependencies that conflict with fedbiomed, so it should be installed "
        "in a separate environment."
    )

from fl_pd.utils.freesurfer import fs_to_pcn
from fl_pd.utils.constants import (
    COLS_PHENO,
    PCN_MODEL_INFO_A2009S_ASEG,
    PCN_MODEL_INFO_DK,
)


def _get_all_site_ids(
    dpath_normative_modelling_data: Path,
    model_info_list: Iterable[Tuple[str, str]],
) -> List[str]:
    site_ids = []
    for _, fname_site_ids_train in model_info_list:
        fpath_training_sites = dpath_normative_modelling_data / fname_site_ids_train
        site_ids_train = fpath_training_sites.read_text().splitlines()
        site_ids.extend(site_ids_train)
    return site_ids


def _get_model_info(
    idp_name: str,
    dpath_normative_modelling_data: Path,
    model_info_list: Iterable[Tuple[str, str]],
) -> Tuple[Path, List[str]] | None:
    for model_info in model_info_list:
        dname_model, fname_site_ids_train = model_info
        dpath_model = dpath_normative_modelling_data / dname_model / idp_name / "Models"
        if dpath_model.exists():
            return dpath_model, _get_all_site_ids(
                dpath_normative_modelling_data=dpath_normative_modelling_data,
                model_info_list=[model_info],
            )
    return None


def get_z_scores(
    df_full: pd.DataFrame,
    df_adaptation: pd.DataFrame,
    dpath_normative_modelling_data: Path | str,
    cols_cov=("AGE", "SEX"),
    xmin=-5,
    xmax=110,
    model_info_list: Iterable[Tuple[str, str]] = (
        PCN_MODEL_INFO_A2009S_ASEG,
        PCN_MODEL_INFO_DK,
    ),
    site_id="NEW",
) -> pd.DataFrame:
    col_site = "site"
    col_sitenum = "sitenum"

    dpath_normative_modelling_data = Path(dpath_normative_modelling_data).resolve()

    if set(df_full.columns) != set(df_adaptation.columns):
        raise ValueError("The columns of df_full and df_adaptation do not match")

    site_ids_all = _get_all_site_ids(
        dpath_normative_modelling_data=dpath_normative_modelling_data,
        model_info_list=model_info_list,
    )

    # treat entire dataframe as new site
    df_full = fs_to_pcn(df_full.copy())
    cols_orig = df_full.columns
    df_adaptation = fs_to_pcn(df_adaptation.copy())
    if site_id in site_ids_all:
        raise ValueError(
            f"Site ID '{site_id}' is already in the training sites. "
            "Please choose a different site ID."
        )
    sitenum = len(site_ids_all) + 1
    df_full[col_site] = site_id
    df_full[col_sitenum] = sitenum
    df_adaptation[col_site] = site_id
    df_adaptation[col_sitenum] = sitenum

    idps_success = []
    with tempfile.TemporaryDirectory(dir=dpath_normative_modelling_data) as dpath_tmp:
        with working_directory(dpath_tmp):
            for idp_name in df_full.columns:

                if (
                    (idp_name in COLS_PHENO)
                    or (idp_name in cols_cov)
                    or (idp_name in (col_site, col_sitenum))
                ):
                    continue

                dpath_model, site_ids_train = _get_model_info(
                    idp_name, dpath_normative_modelling_data, model_info_list
                )
                if dpath_model is None:
                    warnings.warn(
                        f"Model directory {dpath_model} does not exist, skipping"
                    )
                    continue

                # get NA rows
                idx_notna_full = (
                    ~df_full.loc[:, [idp_name] + list(cols_cov)]
                    .isna()
                    .any(axis="columns")
                )
                idx_notna_adaptation = (
                    ~df_adaptation.loc[:, [idp_name] + list(cols_cov)]
                    .isna()
                    .any(axis="columns")
                )

                df_full.loc[idx_notna_full, idp_name] = _get_z_scores_for_idp(
                    df_full=df_full.loc[idx_notna_full],
                    df_adaptation=df_adaptation.loc[idx_notna_adaptation],
                    idp_name=idp_name,
                    dpath_model=dpath_model,
                    dpath_work=dpath_tmp,
                    site_ids_train=site_ids_train,
                    cols_cov=cols_cov,
                    xmin=xmin,
                    xmax=xmax,
                )

                idps_success.append(idp_name)

    print(f"Successfully computed z-scores for {len(idps_success)} IDPs")

    return df_full.loc[:, cols_orig]


def _get_z_scores_for_idp(
    df_full: pd.DataFrame,
    df_adaptation: pd.DataFrame,
    idp_name: str,
    dpath_model: Path,
    site_ids_train: list[str],
    dpath_work: Optional[Path | str] = None,
    cols_cov=("AGE", "SEX"),
    xmin=-5,
    xmax=110,
):
    if dpath_work is None:
        dpath_work = Path(".")
    else:
        dpath_work = Path(dpath_work)

    # extract and save the response variables for the test set
    y_te = df_full[idp_name].to_numpy()

    # save the variables
    resp_file_te = os.path.join(dpath_work, "resp_te.txt")
    np.savetxt(resp_file_te, y_te)

    # configure and save the design matrix
    cov_file_te = os.path.join(dpath_work, "cov_bspline_te.txt")
    X_te = create_design_matrix(
        df_full.loc[:, cols_cov],
        site_ids=df_full["site"],
        all_sites=site_ids_train,
        basis="bspline",
        xmin=xmin,
        xmax=xmax,
    )
    np.savetxt(cov_file_te, X_te)

    # save the covariates for the adaptation data
    X_ad = create_design_matrix(
        df_adaptation.loc[:, cols_cov],
        site_ids=df_adaptation["site"],
        all_sites=site_ids_train,
        basis="bspline",
        xmin=xmin,
        xmax=xmax,
    )
    cov_file_ad = os.path.join(dpath_work, "cov_bspline_ad.txt")
    np.savetxt(cov_file_ad, X_ad)

    # save the responses for the adaptation data
    resp_file_ad = os.path.join(dpath_work, "resp_ad.txt")
    y_ad = df_adaptation[idp_name].to_numpy()
    np.savetxt(resp_file_ad, y_ad)

    # save the site ids for the adaptation data
    sitenum_file_ad = os.path.join(dpath_work, "sitenum_ad.txt")
    site_num_ad = df_adaptation["sitenum"].to_numpy(dtype=int)
    np.savetxt(sitenum_file_ad, site_num_ad)

    # save the site ids for the test data
    sitenum_file_te = os.path.join(dpath_work, "sitenum_te.txt")
    site_num_te = df_full["sitenum"].to_numpy(dtype=int)
    np.savetxt(sitenum_file_te, site_num_te)

    _, _, z_scores = predict(
        cov_file_te,
        alg="blr",
        respfile=resp_file_te,
        model_path=dpath_model,
        adaptrespfile=resp_file_ad,
        adaptcovfile=cov_file_ad,
        adaptvargroupfile=sitenum_file_ad,
        testvargroupfile=sitenum_file_te,
        inputsuffix="estimate",
    )
    return z_scores


@contextmanager
def working_directory(dpath):
    dpath_old = Path.cwd()
    os.chdir(dpath)
    try:
        yield
    finally:
        os.chdir(dpath_old)
