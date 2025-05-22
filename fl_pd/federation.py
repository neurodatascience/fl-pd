from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_initial_params(model, n_features, n_targets, classes=None) -> dict[str, Any]:

    if isinstance(model, Ridge):
        # shape (n_features,) or (n_targets, n_features)
        return {
            "coef_": np.zeros(
                (n_features,) if n_targets == 1 else (n_targets, n_features)
            ),
            "intercept_": np.zeros((n_targets,)),
        }
    elif isinstance(model, LogisticRegression):
        # shape (1, n_features) or (n_classes, n_features)
        if classes is None:
            raise ValueError("classes must be provided for LogisticRegression")
        return {
            "coef_": np.zeros((1, n_features)),
            "intercept_": np.zeros((1,)),
            "classes_": classes,
        }
    elif isinstance(model, StandardScaler):
        return {
            "mean_": np.zeros(n_features),
            "scale_": np.ones(n_features),
            "var_": np.ones(n_features),
            "n_samples_seen_": 0,
        }
    elif isinstance(model, Pipeline):
        params = {}
        for step_name, step_model in model.steps:
            params.update(
                {
                    f"{step_name}__{k}": v
                    for k, v in get_initial_params(
                        step_model, n_features, n_targets, classes=classes
                    ).items()
                }
            )
        return params


def get_fitted_params(model) -> dict[str, Any]:
    if isinstance(model, Pipeline):
        params = {}
        for step_name, step_model in model.steps:
            params.update(
                {
                    f"{step_name}__{k}": v
                    for k, v in get_fitted_params(step_model).items()
                }
            )
        return params
    else:
        if isinstance(model, (Ridge, LogisticRegression)):
            param_names = ["coef_", "intercept_"]
        elif isinstance(model, StandardScaler):
            param_names = ["mean_", "scale_"]
        return {name: getattr(model, name) for name in param_names}


def set_params(model, params: dict[str, Any]):
    if isinstance(model, Pipeline):
        for key, value in params.items():
            step_name, param_name = key.split("__", maxsplit=1)
            set_params(model.named_steps[step_name], {param_name: value})
    else:
        for key, value in params.items():
            setattr(model, key, value)


def average_params(params_list: list[dict], weights=None) -> dict[str, Any]:
    n_models = len(params_list)
    if n_models < 2:
        return params_list

    averaged_params = {}
    for key in params_list[0]:
        original_shape = params_list[0][key].shape
        aggregated_params = np.vstack([params[key] for params in params_list])
        averaged_params[key] = np.reshape(
            np.average(aggregated_params, axis=0, weights=weights),
            original_shape,
        )

    return averaged_params
