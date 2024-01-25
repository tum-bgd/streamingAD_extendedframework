import numpy as np
from scipy.stats import norm
from scipy import special

from abstractAnomalyScore import AbstractAnomalyScore
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from ..nonconformity_scores.euclidean_distance import EuclideanDistanceNonConformity


class ConfidenceLevels(AbstractAnomalyScore):
    def __init__(self, publisher: EuclideanDistanceNonConformity, training_set_publisher: AbstractTrainingSetUpdateMethod,
                 subscribers: list, initial_nonconformity_scores: np.ndarray, confidence_window_length: int) -> None:
        self.publisher = publisher
        self.training_set_publisher = training_set_publisher
        self.subscribers = subscribers
        self.confidence_window_length = confidence_window_length
        self.confidence_window = []
        self.alpha_window = []
        self.anomaly_window = []
        self.training_set_length = self.training_set_publisher.get_window_length()
        assert len(initial_nonconformity_scores) == self.training_set_length
        self.nonconformity_scores_training_set = initial_nonconformity_scores
        self.uniform_cdf_x = None

    def update_parameters(self):
        d = self.training_set_publisher.get_last_added_removed()
        to_remove = d['last_removed_indices'][0]
        to_add = self.publisher.nonconformity_score
        self.nonconformity_scores_training_set.remove(to_remove)
        self.nonconformity_scores_training_set = np.concatenate(
            [self.nonconformity_scores_training_set, to_add])

    def calculate_anomaly_score(self):
        self.current_confidence_level = len(
            self.publisher.nonconformity_score > self.nonconformity_scores_training_set)
        self.current_confidence_level /= self.training_set_length
        if len(self.confidence_window) == self.confidence_window_length:
            self.confidence_window.pop(0)
        self.confidence_window.append(self.current_confidence_level)
        empirical_cdf_x, y = self.calc_cdf(self.confidence_window)
        if self.uniform_cdf_x == None:
            self.uniform_cdf_x = norm.cdf(np.linspace(
                np.min(empirical_cdf_x), np.max(empirical_cdf_x), len(empirical_cdf_x)))
        test_statistic = np.max(np.abs(empirical_cdf_x - self.uniform_cdf_x))

        # Find significance level based on test_statistic = critical_value
        # Smaller alpha -> higher support of being anomalous
        alpha = 2 / np.exp((len(self.confidence_window)/2)
                           * np.square(test_statistic))
        if len(self.alpha_window) == self.confidence_window_length:
            self.alpha_window.pop(0)
        self.alpha_window.append(alpha)
        
        # logarithmic inversion for regularization, min-max scaling for normalization and Gaussian scaling to produce anomaly scores
        # due to negative logarithm: Higher log_inv_alpha -> higher support of being anomalous
        alpha_significant = 0.05
        log_inv_alpha = - np.log(alpha / alpha_significant)
        min_max_alpha = (log_inv_alpha - min(self.alpha_window)) / \
            (max(self.alpha_window) - min(self.alpha_window))
        self.anomaly_score = np.maximum(0, special.erf(
            (min_max_alpha - np.mean(self.alpha_window)) / (np.sqrt(2) * np.std(self.alpha_window))))
        
        if len(self.anomaly_window) == self.confidence_window_length:
            self.anomaly_window.pop(0)
        self.anomaly_window.append(self.anomaly_score)

    def save_anomaly_score(self):
        with open(self.save_path, 'a') as f:
            f.write(self.anomaly_score)

    def notify(self):
        self.update_parameters()
        self.calculate_anomaly_score()
        self.save_anomaly_score()
        for subscriber in self.subscribers:
            subscriber.notify()

    def calc_cdf(self, confidence_levels: np.ndarray):
        y = np.ndarray(range(len(confidence_levels))) / \
            float(len(confidence_levels))
        # this keeps dimension 1 intact
        cdf_x = np.sort(confidence_levels, axis=0)
        cdf_x = (cdf_x - np.mean(cdf_x, axis=0, keepdims=True)) / \
            np.std(cdf_x, axis=0, keepdims=True)
        return (cdf_x, y)
