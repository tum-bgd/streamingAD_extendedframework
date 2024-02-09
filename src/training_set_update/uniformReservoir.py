import numpy as np

from ..dataRepresentation import WindowStreamVectors
from .abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from .slidingWindow import sliding_window

class UniformReservoir(AbstractTrainingSetUpdateMethod):
    def __init__(self, publisher: WindowStreamVectors, reservoir_length: int, first_reservoir: np.ndarray, 
                 subscribers: list, id: str, model_id: str, debug=False) -> None:
        self.publisher:WindowStreamVectors = publisher
        self.id = id
        self.model_id = model_id
        self.reservoir:np.ndarray = first_reservoir
        self.reservoir_length = reservoir_length
        self.feature_vector_length = len(publisher.get_feature_vectors())
        self.subscribers = subscribers
        self.last_removed_indices = []
        self.last_added = None
        self.last_removed = None
        self.debug = debug
    
    def get_training_set(self):
        return self.reservoir
    
    """Update reservoir by removing one of the reservoir values drawn uniformly at a time.
    Note: If this needs to be sped up, remove len(new_values) random elements from reservoir at once.
    """    
    def update_reservoir(self):
        new_feature_vectors = self.publisher.feature_vectors
        step_size = len(new_feature_vectors)
        self.last_added = new_feature_vectors
        to_drop = np.random.randint(0, min(len(self.reservoir), self.reservoir_length), size=(step_size,))
        self.last_removed = self.reservoir[to_drop]
        self.last_removed_indices = to_drop
        self.reservoir = np.concatenate([np.delete(self.reservoir, to_drop, axis=0), new_feature_vectors], axis=0)
        return 0
    
    def get_last_added_removed(self):
        return {
            'last_added': self.last_added,
            'last_removed': self.last_removed,
            'last_removed_indices': self.last_removed_indices,
        }
    
    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        perform_update = np.random.uniform(0, 1) < self.reservoir_length / self.get_update_index()
        if perform_update:
            self.update_reservoir()
            if self.debug:
                print(f'UniformReservoir at {hex(id(self))} update after {time()-t1:.6f}s')
            for subscriber in self.subscribers:
                subscriber.notify()
        else:
            self.last_added = []
            self.last_removed = []
            self.last_removed_indices = []
        return 0
    
    def get_window_length(self) -> int:
        return self.reservoir_length
    
    def get_update_index(self) -> int:
        return self.publisher.publisher.get_update_index()
        
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0
    
    def add_subscribers(self, subscribers):
        self.subscribers.extend(subscribers)
        return 0