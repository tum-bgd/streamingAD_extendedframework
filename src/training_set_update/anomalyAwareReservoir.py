import numpy as np

from ..dataRepresentation import WindowStreamVectors
from .abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from .slidingWindow import sliding_window


""" AnomalyAwareReservoir
This method for updating the training set receives the anomaly score a_t for a new feature vector x_t.
It assumes a higher anomaly score as more "anomalous" and requires a_t in [0,1].
An anomaly score is weighted via exponential decay w_t = exp(-lambda * a_t) and
a random factor u is drawn uniformly from the interval [0,1] to calculate a "priority"
p_t = u^(1/(w_t)).
The reservoir only keeps the top-N feature vectors according to their priorities.
"""


class AnomalyAwareReservoir(AbstractTrainingSetUpdateMethod):
    def __init__(self, publisher: WindowStreamVectors, reservoir_length: int, first_reservoir: np.ndarray,
                 subscribers: list, id: str, model_id: str, update_anomaly_score_params_with_reservoir=False, debug=False) -> None:
        self.publisher: WindowStreamVectors = publisher
        self.anomaly_score_publisher = None
        self.id = id
        self.model_id = model_id
        self.reservoir: np.ndarray = first_reservoir
        self.priorities: np.ndarray = None
        self.reservoir_length = reservoir_length
        self.feature_vector_length = len(publisher.get_feature_vectors())
        self.subscribers = subscribers
        self.last_added = None
        self.last_removed = None
        self.last_removed_indices = []
        self.update_anomaly_score_params_with_reservoir = update_anomaly_score_params_with_reservoir
        self.debug = debug

    def set_anomaly_score_publisher(self, anomaly_score_publisher):
        self.anomaly_score_publisher = anomaly_score_publisher

    def set_first_anomaly_scores(self, first_anomaly_scores):
        assert len(first_anomaly_scores) == self.reservoir_length
        self.priorities = self.calculate_priority(first_anomaly_scores)

    def get_training_set(self):
        return self.reservoir

    """Update reservoir by removing one of the reservoir values drawn uniformly at a time.
    Note: If this needs to be sped up, remove len(new_values) random elements from reservoir at once.
    """

    def update_reservoir(self):
        new_feature_vectors = self.publisher.feature_vectors
        step_size = len(new_feature_vectors)
        new_anomaly_scores = self.anomaly_score_publisher.anomaly_scores
        new_anomaly_scores[~np.isnan(new_anomaly_scores)]
        new_priorities = self.calculate_priority(new_anomaly_scores)
        self.last_added_indices = []
        self.last_removed_indices = []
        updated_any_priority = False
        old_indices = np.arange(len(self.priorities))
        old_priorities_indices = np.stack(
            [self.priorities, old_indices], axis=0)
        sort = np.argsort(old_priorities_indices[0, :])
        for i in range(step_size):
            min_priority = old_priorities_indices[0, sort[0]]
            if min_priority < new_priorities[i] or self.debug:
                to_drop_index = int(old_priorities_indices[1, sort[0]])
                self.last_removed_indices.append(to_drop_index)
                self.last_added_indices.append(i)
                sort = sort[1:]
                updated_any_priority = True
        self.last_added = new_feature_vectors[self.last_added_indices]
        self.last_removed = self.reservoir[self.last_removed_indices]
        self.reservoir = np.concatenate([np.delete(
            self.reservoir, self.last_removed_indices, axis=0), new_feature_vectors[self.last_added_indices]])
        self.priorities = np.concatenate([np.delete(
            self.priorities, self.last_removed_indices, axis=0), new_priorities[self.last_added_indices]])
        if updated_any_priority and self.update_anomaly_score_params_with_reservoir:
            self.anomaly_score_publisher.update_parameters()
        return updated_any_priority

    def get_last_added_removed(self):
        return {
            'last_added': self.last_added,
            'last_added_indices': self.last_added_indices,
            'last_removed': self.last_removed,
            'last_removed_indices': self.last_removed_indices,
        }

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        notify_subscribers = self.update_reservoir()
        if self.debug:
            print(
                f'AnomalyAwareReservoir at {hex(id(self))} update after {time()-t1:.6f}s')
        if notify_subscribers:
            for subscriber in self.subscribers:
                subscriber.notify()
        return 0

    def get_window_length(self) -> int:
        return self.reservoir_length

    def get_update_index(self) -> int:
        return self.publisher.publisher.get_update_index()

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0

    def add_subscribers(self, subscribers: list):
        self.subscribers.extend(subscribers)
        return 0

    def calculate_priority(self, anomaly_scores, lam=3, num=3):
        return np.random.uniform(0.7, 0.9) ** (num / (np.exp(- lam * anomaly_scores)))
