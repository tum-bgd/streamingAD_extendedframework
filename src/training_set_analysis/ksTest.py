import numpy as np

from abstractTrainingSetAnalysisMethod import AbstractTrainingSetAnalysisMethod
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod


class KsTest(AbstractTrainingSetAnalysisMethod):
    def __init__(self, publisher: AbstractTrainingSetUpdateMethod, models: list, alpha: float, check_every=10) -> None:
        self.publisher = publisher
        self.models = models
        self.reset_cumulative_training_set()
        self.alpha = alpha
        self.check_every = check_every
        self.counter = 0

    def reset_cumulative_training_set(self):
        training_set = self.publisher.get_training_set()
        self.cumulative_training_set0 = self.calc_cdf(training_set)
        self.running_training_set_sorted_indices = np.argsort(
            training_set, axis=0)  # this keeps dimension 1 intact
        self.running_mu = np.ndarray = np.mean(training_set, axis=0)
        self.running_var = np.ndarray = np.var(training_set, axis=0)
        self.running_sum = np.sum(training_set, axis=0)
        self.running_sum2 = np.sum(np.square(training_set), axis=0)
        self.counter = 0

    def update_parameters(self):
        d = self.publisher.get_last_added_removed()
        N = self.publisher.get_window_length()
        to_remove = d['last_removed_indices'][0]
        self.running_training_set_sorted_indices.remove(to_remove)
        self.running_training_set_sorted_indices[self.running_training_set_sorted_indices > to_remove] -= 1
        self.running_training_set_sorted_indices = np.concatenate(
            [self.running_training_set_sorted_indices, self.publisher.get_window_length()-1])
        self.running_mu += (1 / N) * (d['last_added'] - d['last_removed'])
        self.running_sum += d['last_added'] - d['last_removed']
        self.running_sum2 += np.square(d['last_added']) - \
            np.square(d['last_removed'])
        self.running_var = (1 / (N-1)) * (self.running_sum2 -
                                          (1/N) * np.square(self.running_sum))

        self.counter += 1
        if self.counter == self.check_every:
            self.check_boundaries()

    def check_boundaries(self):
        running_cdf = self.calc_cdf(self.publisher.get_training_set(), use_indices=True)
        test_statistic = np.mean(running_cdf, axis=1) - np.mean(self.cumulative_training_set0, axis=1)
        test_statistic = np.max(np.abs(test_statistic))
        r = self.publisher.get_window_length()
        critical_value = np.sqrt((2/r) * np.log(2/self.alpha))
        if test_statistic > critical_value:
            self.reset_cumulative_training_set()
            for model in self.models:
                model.retraining(self.publisher.get_training_set())

    def calc_cdf(self, training_set: np.ndarray, use_indices=False):
        y = np.ndarray(range(len(training_set))) / float(len(training_set))
        # this keeps dimension 1 intact
        if use_indices:
            cdf_x = training_set[self.running_training_set_sorted_indices]
        else:
            cdf_x = np.sort(training_set, axis=0)
        cdf_x = (cdf_x - np.mean(cdf_x, axis=0, keepdims=True)) / \
            np.std(cdf_x, axis=0, keepdims=True)
        return (cdf_x, y)

    def notify(self):
        self.update_parameters()
