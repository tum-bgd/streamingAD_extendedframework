import numpy as np

from abstractAnomalyScore import AbstractAnomalyScore
from nonconformity_scores.euclidean_distance import EuclideanDistanceNonConformity


class AverageOfWindow(AbstractAnomalyScore):
    def __init__(self, publisher: EuclideanDistanceNonConformity, subscribers: list, save_path: str, initial_nonconformity_scores: np.ndarray, window_length: int) -> None:
        self.publisher = publisher
        self.subscribers = subscribers
        self.window_length = window_length
        self.save_path = save_path
        self.window = initial_nonconformity_scores[-window_length:]
        self.running_mean = np.mean(initial_nonconformity_scores[-window_length:])

    def update_parameters(self):
        to_add = self.publisher.nonconformity_score
        to_remove = self.window[0]
        self.window = np.concatenate([self.window[1:], to_add])
        self.running_mean += (1/self.window_length) * (to_add - to_remove)

    def calculate_anomaly_score(self):
        self.anomaly_score = self.running_mean

    def save_anomaly_score(self):
        with open(self.save_path, 'a') as f:
            f.write(self.anomaly_score)
    
    def notify(self):
        self.update_parameters()
        self.calculate_anomaly_score()
        self.save_anomaly_score()
        for subscriber in self.subscribers:
            subscriber.notify()
