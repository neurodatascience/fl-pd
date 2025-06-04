def fs6_to_fs7(df_fs6, parcellation=""):

    # aparc
    if parcellation == "DKTatlas":
        df_fs6 = df_fs6.drop(
            columns=["BrainSegVolNotVentSurf", "SupraTentorialVolNotVentVox", "eTIV"],
            errors="ignore",
        )
    elif parcellation == "a2009s":
        df_fs6 = df_fs6.rename(columns=lambda x: x.replace("&", "_and_"))
    elif parcellation == "":
        df_fs6 = df_fs6.drop(
            columns=["BrainSegVolNotVentSurf", "BrainSegVolNotVent", "eTIV"],
            errors="ignore",
        )
    else:
        raise ValueError(f"Invalid parcellation: {parcellation}")

    # aseg
    df_fs6 = df_fs6.rename(
        columns={
            "Left-Thalamus-Proper": "Left-Thalamus",
            "Right-Thalamus-Proper": "Right-Thalamus",
        }
    )

    return df_fs6


def fs_to_pcn(df_fs, parcellation=""):
    def _fs_to_pcn_DK(idp_name: str):
        if not idp_name.endswith("MeanThickness_thickness"):
            idp_name = idp_name.removesuffix("_thickness")
            if idp_name.startswith("lh_"):
                idp_name = f"L_{idp_name.removeprefix('lh_')}"
            elif idp_name.startswith("rh_"):
                idp_name = f"R_{idp_name.removeprefix('rh_')}"
        return idp_name

    # aparc
    if parcellation == "a2009s":
        # FS7 to FS6
        df_fs = df_fs.rename(columns=lambda x: x.replace("_and_", "&"))
    elif parcellation == "":
        # PCN naming is different from FS6?
        df_fs = df_fs.rename(columns=_fs_to_pcn_DK)
    else:
        raise ValueError(f"Invalid parcellation: {parcellation}")

    # aseg
    df_fs = df_fs.rename(
        columns={
            "Left-Thalamus": "Left-Thalamus-Proper",
            "Right-Thalamus": "Right-Thalamus-Proper",
        }
    )
    return df_fs


def fs7_aparc_to_keep(df_fs7, parcellation=""):
    if parcellation == "DKTatlas":
        df_fs7 = df_fs7.drop(
            columns=[
                "lh_MeanThickness_thickness",
                "rh_MeanThickness_thickness",
                "rh_temporalpole_thickness",
            ],
        )
    elif parcellation == "a2009s":
        pass
    elif parcellation == "":
        pass
    else:
        raise ValueError(f"Invalid parcellation: {parcellation}")
    return df_fs7


def fs7_aseg_to_keep(df_fs7):
    cols = [
        "EstimatedTotalIntraCranialVol",
        "Left-Lateral-Ventricle",
        "Left-Thalamus",
        "Left-Caudate",
        "Left-Putamen",
        "Left-Pallidum",
        "Left-Hippocampus",
        "Left-Amygdala",
        "Left-Accumbens-area",
        "Right-Lateral-Ventricle",
        "Right-Thalamus",
        "Right-Caudate",
        "Right-Putamen",
        "Right-Pallidum",
        "Right-Hippocampus",
        "Right-Amygdala",
        "Right-Accumbens-area",
    ]
    return df_fs7[cols]
