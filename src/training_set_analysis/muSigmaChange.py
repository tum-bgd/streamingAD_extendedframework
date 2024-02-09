import numpy as np

from .abstractTrainingSetAnalysisMethod import AbstractTrainingSetAnalysisMethod
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from ..anomaly_scores.averageOfWindow import AverageOfWindow
from ..anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from ..anomaly_scores.confidenceLevels import ConfidenceLevels
from .saveWeightsPath import save_weights_paths


class MuSigmaChange(AbstractTrainingSetAnalysisMethod):
    def __init__(self, publisher: AbstractTrainingSetUpdateMethod, models: list, out_base_path: str, date_id: str, check_every=10, debug=False) -> None:
        self.publisher = publisher
        self.models = models
        self.reset_all_mu_sigma()
        self.check_every = check_every
        self.save_weights_after_retraining = True
        self.out_base_path = out_base_path
        self.date_id = date_id
        self.counter = 0
        self.debug = debug

    def reset_all_mu_sigma(self):
        training_set = self.publisher.get_training_set()
        self.mu0: np.ndarray = np.mean(training_set, axis=0)
        self.var0: np.ndarray = np.var(training_set, axis=0)
        self.running_mu = self.mu0
        self.running_var = self.var0
        self.running_sum = np.sum(training_set, axis=0)
        self.running_sum2 = np.sum(np.square(training_set), axis=0)
        # self.counter = 0

    def update_parameters(self):
        d = self.publisher.get_last_added_removed()
        N = self.publisher.get_window_length()
        self.running_mu += (1 / N) * (np.sum(d['last_added'], axis=0) - np.sum(d['last_removed'], axis=0))
        self.running_sum += np.sum(d['last_added']) - np.sum(d['last_removed'])
        self.running_sum2 += np.square(np.sum(d['last_added'])) - \
            np.square(np.sum(d['last_removed']))
        self.running_var = (1 / (N-1)) * (self.running_sum2 -
                                          (1/N) * np.square(self.running_sum))
        self.counter += 1
        if self.counter % self.check_every == 0:
            self.check_boundaries()

    def check_boundaries(self):
        std0 = np.linalg.norm(self.calc_std(self.var0))
        running_std = np.linalg.norm(self.calc_std(self.running_var))
        check_mu: bool = np.linalg.norm(self.running_mu - self.mu0) > std0
        check_std: bool = (
            (1/2) * std0 > running_std or running_std > 2 * std0)
        if check_mu or check_std or self.debug:
            print(
                f'Model retraining after {self.publisher.get_update_index()} steps - will be reset')
            print(f'Publisher id: {self.publisher.id} - Model id: {self.publisher.model_id}')
            print(
                f'Training set {self.publisher} changed according to mu/sigma')
            publisher_model_id = self.publisher.model_id
            model_id_set = set([x['object'].model_id for x in self.models if x['object'].model_id != 'pcb_iforest'])
            if publisher_model_id == 'all':
                for model_id in model_id_set:
                    self.retrain_model_and_rearrange_variables_tree(
                        model_id=model_id)
            else:
                model_id = publisher_model_id
                self.retrain_model_and_rearrange_variables_tree(
                    model_id=model_id)
            self.reset_all_mu_sigma()

    def retrain_model_and_rearrange_variables_tree(self, model_id):
        publisher_id = self.publisher.id
        training_set = self.publisher.get_training_set()
        model_versions = [(i, x['version']) for i, x in enumerate(
            self.models) if model_id == x['object'].model_id]
        anomaly_score_id = None
        if publisher_id in ['ares_al_mu_sig', 'ares_al_ks'] or model_id == 'pcb_iforest':
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
                    filename=f'{model_id}-musigma-step{self.publisher.get_update_index()}.h5',
                )
                for save_path in save_paths:
                    self.models[model_idx]['object'].save_model(save_path=save_path)
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
            if publisher_id in ['sw_mu_sig', 'ures_mu_sig']:
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
                    filename=f'{model_id}-musigma-step{self.publisher.get_update_index()}.h5',
                )
                for save_path in save_paths:
                    model['object'].save_model(save_path=save_path)

    def calc_std(self, var: np.ndarray):
        return np.sqrt(np.clip(var, 1e-6, 1e+20))

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        self.update_parameters()
        if self.debug:
            print(
                f'MuSigmaChange at {hex(id(self))} update after {time()-t1:.6f}s')
