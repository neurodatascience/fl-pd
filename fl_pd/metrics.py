import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    balanced_accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)

from fl_pd.utils.constants import MlProblem


def get_metrics_map(problem: MlProblem):
    if problem == MlProblem.CLASSIFICATION:
        return {
            "balanced_accuracy": balanced_accuracy_score,
        }
    elif problem == MlProblem.REGRESSION:
        return {
            "r2": r2_score,
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "explained_variance": explained_variance_score,
            "corr": (lambda x, y: pearsonr(np.squeeze(x), np.squeeze(y))[0]),
        }
    raise ValueError(f"No metrics found for {problem=}")
