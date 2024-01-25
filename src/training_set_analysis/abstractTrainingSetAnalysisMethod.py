from abc import abstractmethod
import numpy as np

from ..abstractSubscriber import AbstractSubscriber

class AbstractTrainingSetAnalysisMethod(AbstractSubscriber):
    @abstractmethod
    def notify():
        pass
    
    @abstractmethod
    def update_parameters():
        pass
    
    @abstractmethod
    def check_boundaries():
        pass
