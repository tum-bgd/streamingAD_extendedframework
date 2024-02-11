import time
import numpy as np
import eif as iso
import random
from typing_extensions import override
from copy import deepcopy
import pickle
from typing import Callable

from modelWrapper import ModelWrapper


class PCBIForest(ModelWrapper):
    def __init__(self, n_trees=200, *args, **kwargs) -> None:
        super().__init__(tf_model=None, *args, **kwargs)
        self.n_trees = n_trees
        self.model = None
        self.performance_counters = np.zeros((n_trees,))

    @override
    def train(self, x_train: np.ndarray, epochs):
        assert len(x_train.shape) == 3
        x_train = x_train.reshape(-1, x_train.shape[-1])
        subsample_size = max(10, min(len(x_train)//10, 256))
        self.model = iso.iForest(
            x_train, ntrees=self.n_trees, sample_size=subsample_size, 
            ExtensionLevel=x_train.shape[-1]-1)

    @override
    def retraining(self, training_set: np.ndarray):
        assert len(training_set) == 3
        training_set = training_set.reshape(-1, training_set.shape[-1])
        for j in range(self.n_trees):
            if self.performance_counters[j] < 0:
                self.model.rebuild_tree(training_set, j)
            self.performance_counters[j] = 0

    @override
    def predict_current(self):
        self.current_feature_vectors = self.publisher.feature_vectors
        assert len(self.current_feature_vectors.shape) == 3
        n_samples = len(self.current_feature_vectors)
        self.current_depths = np.zeros((n_samples, self.n_trees), dtype=np.float32)
        for i in range(len(self.current_feature_vectors)):
            for j in range(self.n_trees):
                self.current_depths[i, j] = np.mean(self.model.compute_paths_single_tree(self.current_feature_vectors[i], j))
        self.current_scores = 2.0**(-self.current_depths/self.model.limit)
        self.current_predictions = np.mean(self.current_scores, axis=1)
        
    @override
    def predict(self, x):
        assert len(x.shape) == 3
        result = np.zeros((len(x)))
        for i in range(len(x)):
            result[i] = 2.0**(-np.mean(self.model.compute_paths(x[i]))/self.model.limit)
        return result
        
    def update_performance_counters(self, anomaly_score_fn: Callable, anomaly_threshold: float):
        individual_scores = self.current_scores
        individual_anomaly_scores = anomaly_score_fn(individual_scores)
        anomaly_scores = anomaly_score_fn(self.current_predictions)
        for i in range(len(anomaly_scores)):
            for j in range(self.n_trees):
                if anomaly_scores[i] > anomaly_threshold:
                    if individual_anomaly_scores[i, j] > anomaly_threshold:
                        self.performance_counters[j] += 1
                    else:
                        self.performance_counters[j] -= 1
                else:
                    if individual_anomaly_scores[i, j] > anomaly_threshold:
                        self.performance_counters[j] -= 1
                    else:
                        self.performance_counters[j] += 1
                    
    @override
    def factory_copy(self):
        new_instance = PCBIForest(
            publisher=self.publisher, 
            subscribers=self.subscribers.copy(),
            n_trees=self.n_trees,
            model_id=self.model_id, 
            model_type=self.model_type, 
            debug=self.debug)
        new_instance.model = deepcopy(self.model)
        return new_instance
    
    @override
    def save_model(self, save_path):
        pass
                   
