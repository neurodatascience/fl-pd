import numpy as np
import pandas as pd

from fl_pd.utils.constants import THRESHOLD_MOCA_RATE, THRESHOLD_MMSE_RATE


def cog_decline_from_moca_rate(moca_rate):
    return moca_rate <= THRESHOLD_MOCA_RATE if not np.isnan(moca_rate) else np.nan


def cog_decline_from_mmse_rate(mmse_rate):
    return mmse_rate <= THRESHOLD_MMSE_RATE if not np.isnan(mmse_rate) else np.nan


def cog_decline_from_pad_mci(df_mci: pd.DataFrame) -> pd.DataFrame:
    data_cog_decline = []
    for participant_id, df_mci_participant in df_mci.groupby("participant_id"):
        df_mci_participant = df_mci_participant.sort_values(
            "Candidate_Age", ascending=True
        )
        mci_progression = "".join(df_mci_participant["RC_MCI"].astype(str))
        if "1" in mci_progression:
            n_months_switch = (
                df_mci_participant["visit_id"]
                .str.extract(r"(\d+)")
                .astype(int)
                .iloc[mci_progression.index("1")]
            )
        else:
            n_months_switch = pd.NA
        data_cog_decline.append(
            {
                "participant_id": participant_id,
                "COG_DECLINE": (
                    "01" in mci_progression and "10" not in mci_progression
                ),
                "n_months_switch": n_months_switch,
            }
        )
    df_cog_decline = pd.DataFrame(data_cog_decline).set_index("participant_id")
    return df_cog_decline
