import numpy as np
from scipy.stats import kstest

from .abstractTrainingSetAnalysisMethod import AbstractTrainingSetAnalysisMethod
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from ..training_set_update.anomalyAwareReservoir import AnomalyAwareReservoir
from ..anomaly_scores.averageOfWindow import AverageOfWindow
from ..anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from ..anomaly_scores.confidenceLevels import ConfidenceLevels
from .muSigmaChange import MuSigmaChange
from .saveWeightsPath import save_weights_paths

""" KsTest Training Set Analysis

This training set analysis method keeps the training_set after a model training session stored in memory as a reference.
It compares the current training_set to the reference training set via the two-sided KS test and triggers
model retraining if the test statistic surpasses a predefined critical value.
"""


class KsTest(AbstractTrainingSetAnalysisMethod):
    def __init__(self, publisher: AbstractTrainingSetUpdateMethod, models: dict, alpha: float, out_base_path: str, date_id: str, check_every=50, debug=False) -> None:
        self.publisher = publisher
        self.models = models
        self.reset_cumulative_training_set()
        self.alpha = alpha
        self.check_every = check_every
        self.save_weights_after_retraining = True
        self.out_base_path = out_base_path
        self.date_id = date_id
        self.counter = 0
        self.debug = debug

    def reset_cumulative_training_set(self):
        training_set = self.publisher.get_training_set()
        self.training_set0 = training_set
        # self.counter = 0

    def update_parameters(self):
        self.counter += 1
        if self.counter % self.check_every == 0:
            self.check_boundaries()

    def check_boundaries(self):
        training_set = self.publisher.get_training_set()
        test_statistics = []
        for ch_idx in range(training_set.shape[2]):
            test_results = kstest(training_set[:, :, ch_idx].flatten(
            ), self.training_set0[:, :, ch_idx].flatten())
            test_statistics.append(test_results.statistic)
        r = self.publisher.get_window_length()
        critical_value = np.sqrt((2/r) * np.log(2/self.alpha))
        if any(test_statistics > critical_value) or self.debug:
            print(
                f'Model retraining after {self.publisher.get_update_index()} steps - will be reset')
            print(
                f'Training set at {hex(id(self.publisher))} changed according to KS test with test statistics any({test_statistics} > {critical_value}) (critical value)')
            publisher_model_id = self.publisher.model_id
            model_id_set = set([x['object'].model_id for x in self.models])
            if publisher_model_id == 'all':
                for model_id in model_id_set:
                    self.retrain_model_and_rearrange_variables_tree(
                        model_id=model_id)
            else:
                model_id = publisher_model_id
                self.retrain_model_and_rearrange_variables_tree(
                    model_id=model_id)
            self.reset_cumulative_training_set()

    def retrain_model_and_rearrange_variables_tree(self, model_id):
        publisher_id = self.publisher.id
        training_set = self.publisher.get_training_set()
        model_versions = [(i, x['version']) for i, x in enumerate(
            self.models) if model_id == x['object'].model_id]
        anomaly_score_id = None
        if publisher_id in ['ares_al_mu_sig', 'ares_al_ks']:
            anomaly_score_id = 'anomaly_likelihood'
        elif publisher_id in ['ares_cl_mu_sig', 'ares_cl_ks']:
            anomaly_score_id = 'confidence_levels'
        if publisher_id in [x[1] for x in model_versions]:
            model_idx = [x[0]
                         for x in model_versions if x[1] == publisher_id][0]
            self.models[model_idx]['object'].retraining(training_set)
            if self.save_weights_after_retraining:
                save_paths = save_weights_paths(
                    out_base_path=self.out_base_path,
                    date_id=self.date_id,
                    model_id=model_id,
                    r_id=publisher_id,
                    anomaly_score_id=anomaly_score_id,
                    filename=f'{model_id}-step{self.publisher.get_update_index()}.h5',
                )
                for save_path in save_paths:
                    self.models[model_idx]['object'].tf_model.save_weights(
                        save_path)
        else:
            # Create new branch in variables tree
            model_idx = [x[0]
                         for x in model_versions if x[1] == 'constant_weights'][0]
            model = self.models[model_idx]
            new_branch = {
                "object": model['object'].factory_copy(),
                "version": publisher_id
            }
            data_representation = model['object'].publisher
            data_representation.add_subscriber(new_branch['object'])
            new_branch['nonconformity_score'] = {
                "object": model['nonconformity_score']['object'].factory_copy()}
            new_branch['object'].subscribers = [
                new_branch['nonconformity_score']['object']]
            new_branch['nonconformity_score']['object'].publisher = new_branch['object']
            for save_path in new_branch['nonconformity_score']['object'].save_paths.copy():
                if publisher_id in save_path:
                    if save_path in model['nonconformity_score']['object'].save_paths:
                        model['nonconformity_score']['object'].save_paths.remove(save_path)
                else:
                    new_branch['nonconformity_score']['object'].save_paths.remove(save_path)
            new_branch_anomaly_scores = []
            if publisher_id in ['sw', 'ures']:
                for anomaly_score in model['nonconformity_score']['object'].subscribers:
                    as_copy = anomaly_score.factory_copy()
                    as_copy.publisher = new_branch['nonconformity_score']['object']
                    as_copy.subscribers = []
                    for save_path in as_copy.save_paths.copy():
                        if publisher_id in save_path:
                            anomaly_score.save_paths.remove(save_path)
                        else:
                            as_copy.save_paths.remove(save_path)
                    if isinstance(anomaly_score, (AnomalyLikelihood, AverageOfWindow)):
                        new_branch_anomaly_scores.append(as_copy)
                    if isinstance(anomaly_score, ConfidenceLevels):
                        if anomaly_score.training_set_publisher.id == publisher_id:
                            new_branch_anomaly_scores.append(as_copy)
            elif publisher_id in ['ares_al_mu_sig', 'ares_al_ks']:
                for anomaly_score in model['nonconformity_score']['object'].subscribers:
                    if isinstance(anomaly_score, AnomalyLikelihood):
                        as_copy = anomaly_score.factory_copy()
                        as_copy.publisher = new_branch['nonconformity_score']['object']
                        subscriber_idx = [j for j, x in enumerate(
                                as_copy.subscribers) if x.id == publisher_id][0]
                        as_copy.subscribers = as_copy.subscribers[subscriber_idx:subscriber_idx+1]
                        as_copy.subscribers[0].set_anomaly_score_publisher(as_copy)
                        for save_path in as_copy.save_paths.copy():
                            if publisher_id in save_path:
                                anomaly_score.save_paths.remove(save_path)
                            else:
                                as_copy.save_paths.remove(save_path)
                        anomaly_score.subscribers.pop(subscriber_idx)
                        new_branch_anomaly_scores.append(as_copy)
            elif publisher_id in ['ares_cl_mu_sig', 'ares_cl_ks']:
                for anomaly_score in model['nonconformity_score']['object'].subscribers:
                    if isinstance(anomaly_score, ConfidenceLevels):
                        if anomaly_score.training_set_publisher.id == publisher_id:
                            # Move the confidence level to new branch
                            new_branch_anomaly_scores.append(anomaly_score)
                            model['nonconformity_score']['object'].subscribers.remove(anomaly_score)
                            model['nonconformity_score']['subscribers'].remove(anomaly_score)
                            anomaly_score.publisher = new_branch['nonconformity_score']['object']
                            for save_path in anomaly_score.save_paths.copy():
                                if not publisher_id in save_path:
                                    anomaly_score.save_paths.remove(save_path)
            new_branch['nonconformity_score']['subscribers'] = new_branch_anomaly_scores
            new_branch['nonconformity_score']['object'].subscribers = new_branch_anomaly_scores
            
            # Adjust save paths
            for save_path in new_branch['nonconformity_score']['object'].save_paths.copy():
                if publisher_id in save_path and save_path in model['nonconformity_score']['object'].save_paths:
                    model['nonconformity_score']['object'].save_paths.remove(save_path)
                else:
                    new_branch['nonconformity_score']['object'].save_paths.remove(save_path)
            for i, new_branch_anomaly_score in enumerate(new_branch['nonconformity_score']['object'].subscribers):
                for save_path in new_branch_anomaly_score.save_paths.copy():
                    if publisher_id in save_path and save_path in model['nonconformity_score']['object'].subscribers[i].save_paths:
                        if new_branch_anomaly_score in model['nonconformity_score']['object'].subscribers:
                            old_index = model['nonconformity_score']['object'].subscribers.index(new_branch_anomaly_score)
                            model['nonconformity_score']['object'].subscribers[old_index].save_paths.remove(save_path)
                            model['nonconformity_score']['subscribers'][old_index].save_paths.remove(save_path)
                    else:
                        new_branch_anomaly_score.save_paths.remove(save_path)
                        new_branch['nonconformity_score']['subscribers'].remove(save_path)

            self.models.append(new_branch)

            # Model retraining
            new_branch['object'].retraining(training_set)
            if self.save_weights_after_retraining:
                save_paths = save_weights_paths(
                    out_base_path=self.out_base_path,
                    date_id=self.date_id,
                    model_id=model_id,
                    r_id=publisher_id,
                    anomaly_score_id=anomaly_score_id,
                    filename=f'{model_id}-step{self.publisher.get_update_index()}.h5',
                )
                for save_path in save_paths:
                    model['object'].tf_model.save_weights(
                        save_path)

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        self.update_parameters()
        if self.debug:
            print(f'KSTest at {hex(id(self))} update after {time()-t1:.6f}s')


def is_sorted(arr):
    return np.all(np.diff(arr) >= 0)
