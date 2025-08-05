import numpy as np

from fl_pd.utils.constants import THRESHOLD_MOCA_RATE, THRESHOLD_MMSE_RATE


def cog_decline_from_moca_rate(moca_rate):
    return moca_rate <= THRESHOLD_MOCA_RATE if not np.isnan(moca_rate) else np.nan


def cog_decline_from_mmse_rate(mmse_rate):
    return mmse_rate <= THRESHOLD_MMSE_RATE if not np.isnan(mmse_rate) else np.nan
