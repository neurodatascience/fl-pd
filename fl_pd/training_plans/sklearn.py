from fedbiomed.common.training_plans import FedSGDClassifier, FedSGDRegressor

from fl_pd.training_plans.mixins import TrainingPlanMixin


class SklearnClassifierTrainingPlan(TrainingPlanMixin, FedSGDClassifier):
    pass


class SklearnRegressorTrainingPlan(TrainingPlanMixin, FedSGDRegressor):
    pass
