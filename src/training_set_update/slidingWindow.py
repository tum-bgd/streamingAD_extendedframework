import numpy as np

from ..dataRepresentation import WindowStreamVectors
from .abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod

class SlidingWindow(AbstractTrainingSetUpdateMethod):
    def __init__(self, publisher: WindowStreamVectors, window_length: int, first_reservoir: np.ndarray, 
                 subscribers: list, id: str, model_id: str, debug=False) -> None:
        self.publisher:WindowStreamVectors = publisher
        self.id = id
        self.model_id = model_id
        self.window_length = window_length
        self.feature_vector_length = len(publisher.get_feature_vector())
        self.reservoir = first_reservoir
        self.subscribers = subscribers
        self.last_added = None
        self.last_removed = None
        self.debug = debug
    
    def get_training_set(self):
        return self.reservoir
    
    def update_training_set(self):
        new_feature_vector = self.publisher.get_feature_vector()
        self.last_added = new_feature_vector
        self.last_removed = self.reservoir[0]
        self.reservoir = np.concatenate([self.reservoir[1:], new_feature_vector.reshape((1, *new_feature_vector.shape))])
    
    def get_last_added_removed(self):
        return {
            'last_added': self.last_added,
            'last_removed': self.last_removed,
            'last_removed_indices': [0],
        }
    
    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        self.update_training_set()
        if self.debug:
            print(f'SlidingWindow at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()
            
    def get_window_length(self) -> int:
        return self.window_length
        
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0
    
    def add_subscribers(self, subscribers):
        self.subscribers.extend(subscribers)
        return 0
    
    def get_update_index(self):
        return self.publisher.publisher.get_update_index()

"""Sliding window helper function
"""
def sliding_window(outer_window: np.ndarray, feature_vector_length: int, training_set_length: int):
        num_feature_vectors = outer_window.shape[0] - feature_vector_length + 1
        num_channels = outer_window.shape[1]
        training_set = np.zeros((num_feature_vectors, feature_vector_length, num_channels))
        for i in range(num_feature_vectors):
            training_set[i] = outer_window[i:i+feature_vector_length]
        training_set = training_set[-training_set_length:]
        return training_set