def fs6_to_fs7(df_fs6, parcellation="a2009s"):

    # aparc
    if parcellation == "DKTatlas":
        df_fs6 = df_fs6.drop(
            columns=["BrainSegVolNotVentSurf", "SupraTentorialVolNotVentVox", "eTIV"],
            errors="ignore",
        )
    elif parcellation == "a2009s":
        df_fs6 = df_fs6.rename(columns=lambda x: x.replace("&", "_and_"))
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


def fs7_to_fs6(df_fs7, parcellation="a2009s"):
    # aparc
    if parcellation == "DKTatlas":
        # FS7 is subset of FS6 for DKTatlas
        pass
    elif parcellation == "a2009s":
        df_fs7 = df_fs7.rename(columns=lambda x: x.replace("_and_", "&"))
    else:
        raise ValueError(f"Invalid parcellation: {parcellation}")

    # aseg
    df_fs7 = df_fs7.rename(
        columns={
            "Left-Thalamus": "Left-Thalamus-Proper",
            "Right-Thalamus": "Right-Thalamus-Proper",
        }
    )
    return df_fs7


def fs7_aparc_to_keep(df_fs7, parcellation="a2009s"):
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
