import numpy as np
from abc import abstractmethod, abstractproperty

from ..abstractSubscriber import AbstractSubscriber

class AbstractAnomalyScore(AbstractSubscriber):
    @abstractmethod
    def calculate_anomaly_score(self):
        pass
    
    @abstractmethod
    def save_anomaly_score(self):
        pass
    
    @abstractproperty
    def anomaly_score(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def notify(self):
        pass