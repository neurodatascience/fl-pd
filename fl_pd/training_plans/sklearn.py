from fedbiomed.common.training_plans import FedSGDClassifier, FedSGDRegressor

from fl_pd.training_plans.mixins import DataLoaderMixin, AdamOptimizerMixin


class SklearnClassifierTrainingPlan(DataLoaderMixin, FedSGDClassifier):
    pass


class SklearnRegressorTrainingPlan(DataLoaderMixin, FedSGDRegressor):
    pass
