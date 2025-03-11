# def ppmi_to_adni(df_ppmi):
#     df_ppmi = df_ppmi.rename(columns={
#         "BrainSegVol": "BrainSeg",
#         "BrainSegVolNotVent": "BrainSegNotVent",
#         "CC_Anterior": "CC-Anterior",
#         "CC_Central": "CC-Central",
#         "CC_Mid_Anterior": "CC-Mid-Anterior",
#         "CC_Mid_Posterior": "CC-Mid-Posterior",
#         "CC_Posterior": "CC-Posterior",
#         "EstimatedTotalIntraCranialVol": "EstimatedTotalIntraCranial",
#         "Left-Thalamus": "Left-Thalamus-Proper",
#         "Right-Thalamus": "Right-Thalamus-Proper",
#         "SubCortGrayVol": "SubCortGray",
#         "SupraTentorialVol": "SupraTentorial",
#         "SupraTentorialVolNotVent": "SupraTentorialNotVent",
#         "TotalGrayVol": "TotalGray",
#         "lhCerebralWhiteMatterVol": "Left-CerebralWhiteMatter",
#         "lhCortexVol": "Left-Cortex",
#         "rhCerebralWhiteMatterVol": "Right-CerebralWhiteMatter",
#         "rhCortexVol": "Right-Cortex",
#     })
#     df_ppmi = df_ppmi.drop(
#         columns=[
#             "BrainSegVol-to-eTIV",
#             "CerebralWhiteMatterVol",
#             "CortexVol",
#             "Left-WM-hypointensities",
#             "Left-non-WM-hypointensities",
#             "Mask",
#             "MaskVol-to-eTIV",
#             "Right-WM-hypointensities",
#             "Right-non-WM-hypointensities",
#             "SurfaceHoles",
#             "lhSurfaceHoles",
#             "rhSurfaceHoles",
#             "lh_bankssts_thickness",
#             "lh_frontalpole_thickness",
#             "lh_temporalpole_thickness",
#             "lh_MeanThickness_thickness",
#             "rh_bankssts_thickness",
#             "rh_frontalpole_thickness",
#             "rh_temporalpole_thickness",
#             "rh_MeanThickness_thickness",
#         ],
#         errors='ignore',
#     )
#     return df_ppmi

# def qpn_to_adni(df_qpn):
#     df_qpn = df_qpn.rename(columns={
#         "CC_Anterior": "CC-Anterior",
#         "CC_Central": "CC-Central",
#         "CC_Mid_Anterior": "CC-Mid-Anterior",
#         "CC_Mid_Posterior": "CC-Mid-Posterior",
#         "CC_Posterior": "CC-Posterior",
#         "EstimatedTotalIntraCranialVol": "EstimatedTotalIntraCranial",
#         "Left-Thalamus": "Left-Thalamus-Proper",
#         "Right-Thalamus": "Right-Thalamus-Proper",
#         "VentricleChoroidVol": "VentricleChoroid",
#         "lhCerebralWhiteMatter": "Left-CerebralWhiteMatter",
#         "lhCortex": "Left-Cortex",
#         "rhCerebralWhiteMatter": "Right-CerebralWhiteMatter",
#         "rhCortex": "Right-Cortex",
#     })
#     df_qpn = df_qpn.drop(
#         columns=[
#             "BrainSegVol-to-eTIV",
#             "CerebralWhiteMatter",
#             "Cortex",
#             "Left-WM-hypointensities",
#             "Left-non-WM-hypointensities",
#             "Mask",
#             "MaskVol-to-eTIV",
#             "Right-WM-hypointensities",
#             "Right-non-WM-hypointensities",
#             "SurfaceHoles",
#             "lhSurfaceHoles",
#             "rhSurfaceHoles",
#             "lh_bankssts_thickness",
#             "lh_frontalpole_thickness",
#             "lh_temporalpole_thickness",
#             "lh_MeanThickness_thickness",
#             "rh_bankssts_thickness",
#             "rh_frontalpole_thickness",
#             "rh_temporalpole_thickness",
#             "rh_MeanThickness_thickness",
#         ],
#         errors='ignore',
#     )

#     return df_qpn

# def adni_aseg_to_keep(df_adni):
#     cols = [
#         'EstimatedTotalIntraCranial',
#         'Left-Lateral-Ventricle',
#         'Left-Thalamus-Proper',
#         'Left-Caudate',
#         'Left-Putamen',
#         'Left-Pallidum',
#         'Left-Hippocampus',
#         'Left-Amygdala',
#         'Left-Accumbens-area',
#         'Right-Lateral-Ventricle',
#         'Right-Thalamus-Proper',
#         'Right-Caudate',
#         'Right-Putamen',
#         'Right-Pallidum',
#         'Right-Hippocampus',
#         'Right-Amygdala',
#         'Right-Accumbens-area',
#     ]
#     for col_bad in ["VentricleChoroid", "BrainSegNotVentSurf"]:
#         if col_bad in cols:
#             raise ValueError(f"Cannot include {col_bad}")
#     return df_adni[cols]

# def drop_aseg_cols(df_aseg):
#     return df_aseg.drop(
#         columns=[
#             "Left-WM-hypointensities",
#             "Right-WM-hypointensities",
#             "Left-non-WM-hypointensities",
#             "Right-non-WM-hypointensities",
#             "lhCortexVol",
#             "lhCerebralWhiteMatterVol",
#             "rhCortexVol",
#             "rhCerebralWhiteMatterVol",
#             "BrainSegVolNotVent",
#             "SupraTentorialVolNotVent",
#             "lhSurfaceHoles",
#             "rhSurfaceHoles",
#         ],
#         errors="ignore",
#     )


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
    return df_fs7.drop(columns=[
        'lh_MeanThickness_thickness',
        'rh_MeanThickness_thickness',
        'rh_temporalpole_thickness',
    ])

def fs7_aseg_to_keep(df_fs7):
    cols = [
        'EstimatedTotalIntraCranialVol', 
        'Left-Lateral-Ventricle',
        'Left-Thalamus',
        'Left-Caudate', 
        'Left-Putamen',
        'Left-Pallidum',
        'Left-Hippocampus',
        'Left-Amygdala', 
        'Left-Accumbens-area',
        'Right-Lateral-Ventricle',
        'Right-Thalamus',
        'Right-Caudate',
        'Right-Putamen',
        'Right-Pallidum',
        'Right-Hippocampus',
        'Right-Amygdala',
        'Right-Accumbens-area'
    ]
    return df_fs7[cols]

# def fs7_to_adni(df_fs7):
#     df_fs7 = df_fs7.drop(
#         columns=[
#             "lh_bankssts_thickness",
#             "lh_frontalpole_thickness",
#             "lh_temporalpole_thickness",
#             "lh_MeanThickness_thickness",
#             "rh_bankssts_thickness",
#             "rh_frontalpole_thickness",
#             "rh_temporalpole_thickness",
#             "rh_MeanThickness_thickness",
#             "BrainSegVol",
#             "BrainSegVol-to-eTIV",
#             "CC_Anterior",
#             "CC_Central",
#             "CC_Mid_Anterior",
#             "CC_Mid_Posterior",
#             "CC_Posterior",
#             "CerebralWhiteMatterVol",
#             "CortexVol",
#             "EstimatedTotalIntraCranialVol",
#             "MaskVol",
#             "MaskVol-to-eTIV",
#             "SubCortGrayVol",
#             "SupraTentorialVol",
#             "SurfaceHoles",
#             "TotalGrayVol",
#             "BrainSeg",
#             "BrainSegNotVent",
#             "Mask",
#             "CerebralWhiteMatter",
#             "Cortex",
#             "SubCortGray",
#             "SupraTentorial",
#             "SupraTentorialNotVent",
#             "TotalGray",
#             "VentricleChoroidVol",
#             "lhCerebralWhiteMatter",
#             "lhCortex",
#             "rhCerebralWhiteMatter",
#             "rhCortex",
#         ],
#         errors="ignore",
#     )
#     # df_fs7 = df_fs7.rename(
#     #     columns={
#     #         "Left-Thalamus": "Left-Thalamus-Proper",
#     #         "Right-Thalamus": "Right-Thalamus-Proper",
#     #     }
#     # )
#     return df_fs7
