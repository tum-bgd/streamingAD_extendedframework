import numpy as np

from ..dataRepresentation import WindowStreamVectors
from abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod

class SlidingWindow(AbstractTrainingSetUpdateMethod):
    def __init__(self, publisher: WindowStreamVectors, window_length: int, first_reservoir: np.ndarray, subscribers: list) -> None:
        self.publisher:WindowStreamVectors = publisher
        self.window_length = window_length
        self.feature_vector_length = len(publisher.get_feature_vector())
        self.reservoir = first_reservoir
        self.subscribers = subscribers
    
    def get_training_set(self):
        return self.reservoir
    
    def update_training_set(self):
        new_feature_vector = self.publisher.get_feature_vector()
        self.last_added = new_feature_vector.reshape((1, *new_feature_vector.shape))
        self.last_removed = self.reservoir[0:1]
        self.reservoir = np.concatenate([self.reservoir[1:], new_feature_vector])
    
    def get_last_added_removed(self):
        return {
            'last_added': self.last_added,
            'last_removed': self.last_removed,
            'last_removed_indices': [0],
        }
    
    def notify(self):
        self.update_training_set()
        for subscriber in self.subscribers:
            subscriber.notify()
            
    def get_window_length(self) -> int:
        self.window_length

"""Sliding window helper function
"""
def sliding_window(outer_window, feature_vector_length):
        num_feature_vectors = outer_window.shape[0] - feature_vector_length + 1
        num_channels = outer_window.shape[1]
        training_set = np.zeros((num_feature_vectors, feature_vector_length, num_channels))
        for i in range(num_feature_vectors):
            training_set[i] = outer_window[i:i+feature_vector_length]
        return training_set