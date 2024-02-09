import numpy as np
from abc import abstractmethod

from ..abstractSubscriber import AbstractSubscriber
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod

class AbstractAnomalyScore(AbstractSubscriber):
    @abstractmethod
    def save_anomaly_scores(self):
        pass
    
    @abstractmethod
    def get_anomaly_scores(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def add_subscriber(self, subscriber):
        pass
    
    @abstractmethod
    def update_parameters(self):
        pass
    
    @abstractmethod
    def notify(self):
        pass