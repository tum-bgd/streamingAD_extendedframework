from abc import abstractmethod
import numpy as np

from ..abstractSubscriber import AbstractSubscriber

class AbstractTrainingSetUpdateMethod(AbstractSubscriber):
    @abstractmethod
    def notify(self):
        pass
    
    @abstractmethod
    def get_training_set(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_last_added_removed(self) -> "dict[str, np.ndarray]":
        pass
    
    @abstractmethod
    def get_window_length(self) -> int:
        pass
    
    @abstractmethod
    def get_update_index(self) -> int:
        pass
    
    @abstractmethod
    def add_subscriber(self):
        pass
