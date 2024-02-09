import os
import numpy as np
import pandas as pd
from copy import copy
from scipy.stats import uniform, kstest
from scipy import special

from ..tsWindowPublisher import TsWindowPublisher
from .abstractAnomalyScore import AbstractAnomalyScore
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from ..nonconformity_scores.nonconformity_wrapper import NonConformityWrapper

""" ConfidenceLevels
from Calikus, E.; Nowaczyk, S.; SantAnna, A.; Dikmen, O. No Free Lunch but a Cheaper Supper: A General Framework for Streaming Anomaly Detection. 
Expert Systems with Applications 2020, 155, 113453. https://doi.org/10.1016/j.eswa.2020.113453.

This anomaly score method keeps the nonconformity scores corresponding to the instances inside the training set
stored in memory. It calculates a confidence value at every time step, which is defined as
c_t = (1/N) * |nc_i  >= nc_t : i in [1, N]|
--> If few noconformity values are larger than the current, c_t -> 0 (low confidence in "normalness" of sample)
--> If many nc values are larger than the current, c_t -> 1 (high confidence)
Once a new confidence value has been calculated, this method performs the one-sided KS test for every feature channel.
This entails verifying if the past W confidence values are representative of their supposed distribution. 
Thereby, this distribution F(x) does not need to be known, as the test statistic
D_n = sup_{-inf<x<inf}|F_n(x) - F(x)| = sup_{0<=t<=1}|F_n(F^-1(t)) - F(F^-1(t))| 
    = sup_{0<=t<=1}|U_n(t) - t|
does not depend on it. Instead it only depends on the empirical uniform distribution function U_n(t).
--> Suppose, many low confidence values arrive after another --> D_n increases
Under the null hypothesis that the confidence values come from the hypothesized distribution F(x),
K_n = np.sqrt(N) * D_n converges to the Kolmogorov distribution Pr(K), which does not depend on F(x).
With the probability Pr(K <= x) as the CDF of the Kolmogorov distribution (probability that K_n takes on
a value equal or less than x), a goodness-of-fit test can be constructed where the tail probability (p-value) 
is found with
Pr(K <= K_n) = 1 - p-value
for a specific K_n.
For a significance level alpha and a corresponding critical value K_alpha, the null hypothesis can be rejected if
K_n > K_alpha
with Pr(K <= K_alpha) = 1 - alpha

--> If D_n increases, the tail probabilty (p-value) of observing it, according to Kolomgorov dist, decreases

The anomaly score for c_t is then the calculated probability: 1 - p-value
--> If anomaly score decreases (p-value increases), the current data is considered more "normal"
--> If anomaly score increases (p-value decreases), the current data is considered more "anomalous"

(from: Dimitrova, D. S.; Kaishev, V. K.; Tan, S. Computing the Kolmogorov-Smirnov Distribution when the Underlying cdf is Purely Discrete, Mixed or Continuous. 
https://openaccess.city.ac.uk/id/eprint/18541/ (accessed 2024-01-31))
"""
class ConfidenceLevels(AbstractAnomalyScore):
    def __init__(self, publisher: NonConformityWrapper, ts_window_publisher: TsWindowPublisher, training_set_publisher_id: str,
                 subscribers: list, save_paths: "list[str]", initial_nonconformity_scores: np.ndarray, confidence_window_length: int,
                 training_set_length: int, update_parameters_with_notify=True, debug=False) -> None:
        self.publisher = publisher
        self.ts_window_publisher = ts_window_publisher
        self.training_set_publisher_id = training_set_publisher_id
        self.subscribers = subscribers
        self.confidence_window_length = confidence_window_length
        self.confidence_window = []
        self.pvalue_window_length = confidence_window_length // 10
        self.pvalue_window = []
        self.training_set_length = training_set_length
        self.save_paths = save_paths
        self.nonconformity_scores_training_set = initial_nonconformity_scores[-training_set_length:]
        self.uniform_cdf_x = None
        self.update_parameters_with_notify = update_parameters_with_notify
        self.max_neg_log_pvalue = 1
        self.debug = debug

    def set_training_set_publisher(self, training_set_publisher: AbstractTrainingSetUpdateMethod):
        self.training_set_publisher = training_set_publisher

    def update_parameters(self):
        d = self.training_set_publisher.get_last_added_removed()
        if len(d['last_removed_indices']) != 0:
            to_remove_indices = d['last_removed_indices']
            if 'last_added_indices' in d.keys():
                to_add = self.publisher.nonconformity_scores[d['last_added_indices']]
            else:
                to_add = self.publisher.nonconformity_scores
            self.nonconformity_scores_training_set = \
                np.concatenate(
                    [np.delete(self.nonconformity_scores_training_set, to_remove_indices, axis=0), to_add])

    def calculate_anomaly_scores(self):
        step_size = len(self.publisher.nonconformity_scores)
        self.current_confidence_levels = \
            [np.count_nonzero(self.nonconformity_scores_training_set >= self.publisher.nonconformity_scores[i]) / \
            self.training_set_length for i in range(step_size)]
        self.current_pvalues, self.anomaly_scores = np.zeros((step_size,)), np.zeros((step_size,))
        for i, ccl in enumerate(self.current_confidence_levels):
            if len(self.confidence_window) == self.confidence_window_length:
                self.confidence_window.pop(0)
                self.confidence_window.append(ccl)
                unique_cw = set(self.confidence_window)
                if len(unique_cw) > 30:
                    test_results = kstest(list(unique_cw), uniform.cdf)
                    self.current_pvalues[i] = test_results.pvalue
                    if len(self.pvalue_window) == self.pvalue_window_length:
                        self.pvalue_window.pop(0)
                    self.pvalue_window.append(test_results.pvalue)
                    
                    # unification step (neg logarithm, normalization, min-max scaling)
                    neg_log_pvalues = - np.log(np.clip(self.pvalue_window, 1e-40, 1))
                    # normalized_neg_log_pvalues = (neg_log_pvalues - np.mean(neg_log_pvalues, keepdims=True)) / np.clip(np.std(neg_log_pvalues, keepdims=True), 1e-6, 1e+6)
                    # anomaly_scores = (normalized_neg_log_pvalues - np.min(normalized_neg_log_pvalues)) / np.clip(self.max_normalized_neg_log_pvalue - np.min(normalized_neg_log_pvalues), 1e-6, 1e+6)
                    anomaly_scores = (neg_log_pvalues - np.min(neg_log_pvalues)) / np.clip(max(self.max_neg_log_pvalue, np.max(neg_log_pvalues)) - np.min(neg_log_pvalues), 1e-6, 1e+6)
                    self.anomaly_scores[i] = anomaly_scores[-1]
                    if np.max(neg_log_pvalues) > 0.9 * self.max_neg_log_pvalue:
                        self.max_neg_log_pvalue = np.max(neg_log_pvalues) + 5
                    # if self.anomaly_scores[i] > 0.95:
                    #     test = 1
                    #     confidence_window_numpy = np.array(self.confidence_window)
                else:
                    self.current_pvalues[i] = np.nan
                    self.anomaly_scores[i] = np.nan
            else:
                self.confidence_window.append(ccl)
                self.current_pvalues[i] = np.nan
                self.anomaly_scores[i] = np.nan

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
            
    def save_confidence_level(self):
        update_index = self.ts_window_publisher.get_update_index()
        for save_path in self.save_paths:
            parent_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            cl_save_path = '/'.join(save_path.split('/')[:-1] + ['confidence_levels.csv'])
            entry = pd.DataFrame(self.current_confidence_levels, columns=[
                                    'confidence_level'], index=[i for i in range(update_index-len(self.current_confidence_levels)+1, update_index+1)])
            if not os.path.exists(cl_save_path):
                entry.index.name = 'update_index'
                entry.to_csv(cl_save_path, mode='w')
            else:
                entry.to_csv(cl_save_path, mode='a', header=False)
                
    def save_pvalue(self):
        update_index = self.ts_window_publisher.get_update_index()
        for save_path in self.save_paths:
            parent_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            cl_save_path = '/'.join(save_path.split('/')[:-1] + ['pvalues.csv'])
            entry = pd.DataFrame(self.current_pvalues, columns=[
                                    'pvalue'], index=[i for i in range(update_index-len(self.current_pvalues)+1, update_index+1)])
            if not os.path.exists(cl_save_path):
                entry.index.name = 'update_index'
                entry.to_csv(cl_save_path, mode='w')
            else:
                entry.to_csv(cl_save_path, mode='a', header=False)

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        if self.update_parameters_with_notify:
            self.update_parameters()
        self.calculate_anomaly_scores()
        self.save_anomaly_scores()
        self.save_confidence_level()
        self.save_pvalue()
        if self.debug:
            print(f'ConfidenceLevels at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()

    def get_anomaly_scores(self) -> np.ndarray:
        return self.anomaly_scores

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        
    def factory_copy(self):
        new_instance = ConfidenceLevels(
            publisher=self.publisher,
            ts_window_publisher=self.ts_window_publisher,
            training_set_length=self.training_set_length,
            training_set_publisher_id=self.training_set_publisher_id,
            subscribers=copy(self.subscribers),
            save_paths=copy(self.save_paths),
            initial_nonconformity_scores=self.nonconformity_scores_training_set.copy(),
            confidence_window_length=self.confidence_window_length,
            debug=self.debug)
        new_instance.confidence_window = self.confidence_window.copy()
        new_instance.anomaly_scores = self.anomaly_scores
        new_instance.pvalue_window = self.pvalue_window.copy()
        new_instance.update_parameters_with_notify = self.update_parameters_with_notify
        new_instance.training_set_publisher = self.training_set_publisher
        return new_instance
        
