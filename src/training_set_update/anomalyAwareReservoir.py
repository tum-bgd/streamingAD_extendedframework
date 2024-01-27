import numpy as np

from ..dataRepresentation import WindowStreamVectors
from ..anomaly_scores.abstractAnomalyScore import AbstractAnomalyScore
from abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from slidingWindow import sliding_window


class AnomalyAwareReservoir(AbstractTrainingSetUpdateMethod):
    def __init__(self, publisher: WindowStreamVectors, anomalyScorePublisher: AbstractAnomalyScore, reservoir_length: int, 
                 first_reservoir: np.ndarray, first_anomaly_scores: np.ndarray, subscribers: list) -> None:
        self.publisher: WindowStreamVectors = publisher
        self.anomaly_score_publisher = anomalyScorePublisher
        self.reservoir: np.ndarray = first_reservoir
        self.priorities: np.ndarray = self.calculate_priority(first_anomaly_scores)
        self.reservoir_length = reservoir_length
        self.feature_vector_length = len(publisher.get_feature_vector())
        self.base_of_exponent = np.random.normal()
        self.exponent_lambda = np.random.normal()
        self.subscribers = subscribers
        self.last_removed_indices = []

    def get_training_set(self):
        return self.reservoir

    """Update reservoir by removing one of the reservoir values drawn uniformly at a time.
    Note: If this needs to be sped up, remove len(new_values) random elements from reservoir at once.
    """
    def update_reservoir(self):
        new_feature_vector = self.publisher.get_feature_vector()
        new_anomaly_score = self.anomaly_score_publisher.anomaly_score
        new_priority = self.calculate_priority(new_anomaly_score)
        self.last_added = new_feature_vector.reshape(
            (1, *new_feature_vector.shape))
        min_priority = np.min(self.priorities)
        if min_priority < new_priority:
            to_drop_index = np.argmin(self.priorities)
            self.last_removed = self.reservoir[to_drop_index:to_drop_index+1]
            self.last_removed_indices = [to_drop_index]
            self.reservoir = np.append(
                np.delete(self.reservoir, to_drop_index), new_feature_vector)
            self.priorities = np.append(
                np.delete(self.priorities, to_drop_index), new_priority)
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
        
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0

    def calculate_priority(self, anomaly_score):
        return self.base_of_exponent ** (1 / (np.exp(- self.exponent_lambda * anomaly_score)))
