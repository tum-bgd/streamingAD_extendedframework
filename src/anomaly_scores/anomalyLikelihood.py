import numpy as np
from scipy import special

from abstractAnomalyScore import AbstractAnomalyScore
from ..nonconformity_scores.euclidean_distance import EuclideanDistanceNonConformity

class AnomalyLikelihood(AbstractAnomalyScore):
    def __init__(self, publisher: EuclideanDistanceNonConformity, subscribers: list, save_path: str, initial_nonconformity_scores: np.ndarray, short_term_length: int, long_term_length: int) -> None:
        self.publisher = publisher
        self.subscribers = subscribers
        self.save_path = save_path
        self.short_term_length = short_term_length
        self.long_term_length = long_term_length
        self.nonconformity_scores_reservoir = initial_nonconformity_scores[-long_term_length:]
        self.running_short_term_sum = np.sum(self.nonconformity_scores_reservoir[-self.short_term_length:])
        self.running_long_term_sum = np.sum(self.nonconformity_scores_reservoir[-self.long_term_length:])
        self.running_long_term_sum2 = np.sum(np.square(self.nonconformity_scores_reservoir[-self.long_term_length:]))
        
    def update_parameters(self):
        nonconformity_score = self.publisher.nonconformity_score
        self.last_removed = self.nonconformity_scores_reservoir[0:1]
        self.nonconformity_scores_reservoir = np.concatenate([self.nonconformity_scores_reservoir[1:], nonconformity_score])
        self.running_short_term_sum += self.running_short_term_sum + nonconformity_score - self.last_removed
        self.running_long_term_sum += self.running_long_term_sum + nonconformity_score - self.last_removed
        self.running_long_term_sum2 += self.running_long_term_sum2 + np.square(nonconformity_score) - np.square(self.last_removed)
        
    def calculate_anomaly_score(self):
        short_term_mean = (1/self.short_term_length) * self.running_short_term_sum
        long_term_mean = (1/self.long_term_length) * self.running_long_term_sum
        long_term_std = (1 / (self.long_term_length-1)) * (self.running_long_term_sum2 - (1/self.long_term_length) * np.square(self.running_long_term_sum))
        self.anomaly_score = 1 - qfunc((short_term_mean - long_term_mean) / long_term_std)
    
    def save_anomaly_score(self):
        with open(self.save_path, 'a') as f:
            f.write(self.anomaly_score)
    
    def notify(self):
        self.update_parameters()
        self.calculate_anomaly_score()
        self.save_anomaly_score()
        for subscriber in self.subscribers:
            subscriber.notify()
            
def qfunc(x):
    return 0.5 - 0.5 * special.erf(x/np.sqrt(2))