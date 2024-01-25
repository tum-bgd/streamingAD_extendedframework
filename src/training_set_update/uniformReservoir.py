import numpy as np

from ..dataRepresentation import WindowStreamVectors
from abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from slidingWindow import sliding_window

class UniformReservoir(AbstractTrainingSetUpdateMethod):
    def __init__(self, publisher: WindowStreamVectors, reservoir_length: int, first_reservoir: np.ndarray, subscribers: list) -> None:
        self.publisher:WindowStreamVectors = publisher
        self.reservoir:np.ndarray = first_reservoir
        self.reservoir_length = reservoir_length
        self.feature_vector_length = len(publisher.get_feature_vector())
        self.subscribers = subscribers
        self.last_removed_indices = []
    
    def get_training_set(self):
        return self.reservoir
    
    """Update reservoir by removing one of the reservoir values drawn uniformly at a time.
    Note: If this needs to be sped up, remove len(new_values) random elements from reservoir at once.
    """    
    def update_reservoir(self):
        new_feature_vector = self.publisher.get_feature_vector()
        self.last_added = new_feature_vector.reshape((1, *new_feature_vector.shape))
        to_drop = np.random.randint(0, self.reservoir_length)
        self.last_removed = self.reservoir[to_drop:to_drop+1]
        self.last_removed_indices = [to_drop]
        self.reservoir = np.append(np.delete(self.reservoir, to_drop), new_feature_vector)
        return 0
    
    def get_last_added_removed(self):
        return {
            'last_added': self.last_added,
            'last_removed': self.last_removed,
            'last_removed_indices': self.last_removed_indices,
        }
    
    def notify(self):
        self.update_reservoir()
        for subscriber in self.subscribers:
            subscriber.notify()
        return 0
    
    def get_window_length(self) -> int:
        self.reservoir_length