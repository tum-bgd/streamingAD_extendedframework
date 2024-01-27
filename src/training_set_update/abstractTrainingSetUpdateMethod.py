from abc import abstractmethod
import numpy as np

from ..abstractSubscriber import AbstractSubscriber

class AbstractTrainingSetUpdateMethod(AbstractSubscriber):
    @abstractmethod
    def notify():
        pass
    
    @abstractmethod
    def get_training_set() -> np.ndarray:
        pass
    
    @abstractmethod
    def get_last_added_removed() -> dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def get_window_length() -> int:
        pass
    
    @abstractmethod
    def add_subscriber():
        pass
