CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "show_default": True}


def fs6_to_fs7(df_fs6):

    df_fs6 = df_fs6.drop(
        columns=["BrainSegVolNotVentSurf", "SupraTentorialVolNotVentVox", "eTIV"],
        errors="ignore",
    )
    df_fs6 = df_fs6.rename(
        columns={
            "Left-Thalamus-Proper": "Left-Thalamus",
            "Right-Thalamus-Proper": "Right-Thalamus",
        }
    )
    return df_fs6


def fs7_aparc_to_keep(df_fs7):
    return df_fs7.drop(
        columns=[
            "lh_MeanThickness_thickness",
            "rh_MeanThickness_thickness",
            "rh_temporalpole_thickness",
        ]
    )


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
