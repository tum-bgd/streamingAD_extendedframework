import os
import numpy as np
import pandas as pd
from copy import copy
from scipy import special
from scipy.stats import norm

from ..tsWindowPublisher import TsWindowPublisher
from .abstractAnomalyScore import AbstractAnomalyScore
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from ..nonconformity_scores.nonconformity_wrapper import NonConformityWrapper

""" AnomalyLikelihood 
(from: Ahmad, S.; Purdy, S. Real-Time Anomaly Detection for Streaming Analytics. 
arXiv July 8, 2016. http://arxiv.org/abs/1607.02480 (accessed 2024-01-03).)

This anomaly measure keeps track of a short-term mean and long-term mean/std of past nonconformity scores.
The anomaly score is calculated according to:
anomaly_score = norm.cdf((short_term_mean - long_term_mea) / long_term_std)
If short_term_mean > long_term_mean, anomaly_score > 0.5 and increasing to 1 (very anomalous)
If short_term_mean < long_term_mean, anomaly_score < 0.5 and decreasing to 0 (very nominal)
"""


class AnomalyLikelihood(AbstractAnomalyScore):
    def __init__(self, publisher: NonConformityWrapper, ts_window_publisher: TsWindowPublisher, subscribers: list,
                 save_paths: list[str], initial_nonconformity_scores: np.ndarray, short_term_length: int,
                 long_term_length: int, update_at_notify=[], threshold=0.9, debug=False) -> None:
        self.publisher = publisher
        self.ts_window_publisher = ts_window_publisher
        self.subscribers = subscribers
        self.save_paths = save_paths
        self.short_term_length = short_term_length
        self.long_term_length = long_term_length
        self.nonconformity_scores_reservoir = initial_nonconformity_scores[-long_term_length:]
        self.running_short_term_sum = np.sum(
            self.nonconformity_scores_reservoir[-self.short_term_length:])
        self.running_long_term_sum = np.sum(
            self.nonconformity_scores_reservoir[-self.long_term_length:])
        self.running_long_term_sum2 = np.sum(
            np.square(self.nonconformity_scores_reservoir[-self.long_term_length:]))
        self.update_at_notify = update_at_notify
        self.threshold = threshold
        self.debug = debug
        
        np.seterr(all='raise')

    def update_parameters(self):
        nonconformity_scores = self.publisher.nonconformity_scores
        step_size = len(nonconformity_scores)
        L1, L2 = self.short_term_length, self.long_term_length
        self.last_removed = self.nonconformity_scores_reservoir[:step_size]
        last_removed_short_term = np.concatenate([self.nonconformity_scores_reservoir[-L1:], nonconformity_scores[:-L1]])
        last_removed_long_term = np.concatenate([self.nonconformity_scores_reservoir[-L2:], nonconformity_scores[:-L2]])
        short_term_sums = np.array([(self.running_short_term_sum + np.sum(nonconformity_scores[:i]) - np.sum(last_removed_short_term[:i])) for i in range(1, step_size+1)])
        long_term_sums = np.array([(self.running_long_term_sum + np.sum(nonconformity_scores[:i]) - np.sum(last_removed_long_term[:i])) for i in range(1, step_size+1)])
        long_term_sums2 = np.array([(self.running_long_term_sum2 + np.square(np.sum(nonconformity_scores[:i])) - np.square(np.sum(last_removed_long_term[:i]))) for i in range(1, step_size+1)])
        short_term_means = (1/L1) * short_term_sums
        long_term_means = (1/L2) * long_term_sums
        long_term_stds = np.sqrt(np.clip((1 / (L2-1)) * (long_term_sums2 - (
            1/L2) * np.square(np.clip(long_term_sums, 1e-6, 1e+6))), 1e-6, 1e+20))
        self.anomaly_scores = 1 - \
            qfunc((short_term_means - long_term_means) /
                np.clip(long_term_stds, 0.5, None))
        if any(self.anomaly_scores > 0.95):
            test = 1
        self.nonconformity_scores_reservoir = np.concatenate(
            [self.nonconformity_scores_reservoir[step_size:], nonconformity_scores])
        self.running_short_term_sum += np.sum(nonconformity_scores) - np.sum(last_removed_short_term)
        self.running_long_term_sum += np.sum(nonconformity_scores) - np.sum(last_removed_long_term)
        self.running_long_term_sum2 += np.square(
            np.sum(nonconformity_scores)) - np.square(np.sum(last_removed_long_term))

    def calculate_anomaly_scores(self, nonconformity_score):
        # no update in this function
        last_removed = self.nonconformity_scores_reservoir[0]
        short_term_mean = (1/self.short_term_length) * (self.running_short_term_sum + nonconformity_score - last_removed)
        long_term_mean = (1/self.long_term_length) * (self.running_long_term_sum + nonconformity_score - last_removed)
        long_term_std = np.sqrt(np.clip((1 / (self.long_term_length-1)) * ((self.running_long_term_sum2 + np.square(nonconformity_score - last_removed)) - (
            1/self.long_term_length) * np.square(np.clip(self.running_long_term_sum, 1e-6, 1e+6))), 1e-6, 1e+20))
        anomaly_score = 1 - \
            qfunc((short_term_mean - long_term_mean) /
                np.clip(long_term_std, 0.5, None))
        return anomaly_score

    def save_anomaly_scores(self):
        update_index = self.ts_window_publisher.get_update_index()
        for save_path in self.save_paths:
            parent_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            entry = pd.DataFrame(self.anomaly_scores, columns=[
                'anomaly_score'], index=[i for i in range(update_index-len(self.anomaly_scores)+1, update_index+1)])
            if not os.path.exists(save_path):
                entry.index.name = 'update_index'
                entry.to_csv(save_path, mode='w')
            else:
                entry.to_csv(save_path, mode='a', header=False)

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        for fn, args in self.update_at_notify:
            fn(*args)
        self.update_parameters()
        self.save_anomaly_scores()
        if self.debug:
            print(
                f'AnomalyLikelihood at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()

    def get_anomaly_scores(self) -> np.ndarray:
        return self.anomaly_scores

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def factory_copy(self):
        new_instance = AnomalyLikelihood(
            publisher=self.publisher,
            ts_window_publisher=self.ts_window_publisher,
            subscribers=copy(self.subscribers),
            save_paths=copy(self.save_paths),
            initial_nonconformity_scores=self.nonconformity_scores_reservoir.copy(),
            short_term_length=self.short_term_length,
            long_term_length=self.long_term_length,
            debug=self.debug)
        new_instance.anomaly_scores = self.anomaly_scores
        new_instance.last_removed = self.last_removed
        return new_instance
        


def qfunc(x):
    return 0.5 - 0.5 * special.erf(x/np.sqrt(2))
