import numpy as np

from ..abstractSubscriber import AbstractSubscriber
from ..modelWrapper import ModelWrapper


class EuclideanDistanceNonConformity(AbstractSubscriber):
    def __init__(self, publisher: ModelWrapper, subscribers: list) -> None:
        self.subscribers = subscribers
        self.publisher = publisher

    def calc_nonconformity_score(self):
        self.current_feature_vector = self.publisher.current_feature_vector
        self.nonconformity_score = np.linalg.norm(
            self.publisher.current_feature_vector - self.publisher.current_prediction)

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def notify(self):
        self.calc_nonconformity_score()
        for subscriber in self.subscribers:
            subscriber.notify()
