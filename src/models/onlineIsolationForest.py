import time
import numpy as np
import eif as iso
import random
from typing_extensions import override
from copy import deepcopy
import pickle
from typing import Callable

from ..dataRepresentation import WindowStreamVectors
from ..modelWrapper import ModelWrapper


class PCBIForest(ModelWrapper):
    def __init__(self, n_trees=200, concat_to_channel_size=10, *args, **kwargs) -> None:
        super().__init__(tf_model=None, *args, **kwargs)
        self.n_trees = n_trees
        self.model = None
        self.performance_counters = np.zeros((n_trees,))
        self.concat_to_channel_size = concat_to_channel_size

    def _accumulate_sample(self, sample):
        window_length, n_channels = sample.shape
        if self.concat_to_channel_size != None:
            skip_size = max(1, self.concat_to_channel_size // n_channels)
            return [sample[j:j+skip_size, :].flatten().tolist()
                    for j in range(0, window_length-skip_size+1, skip_size)]
        else:
            return [sample[j, :].tolist()
                    for j in range(0, window_length)]

    @override
    def train(self, x: np.ndarray, epochs):
        assert len(x.shape) == 3
        n_samples, window_length, n_channels = x.shape
        x_train = []
        for i in range(n_samples):
            x_train.extend(self._accumulate_sample(x[i]))

        subsample_size = max(10, min(len(x_train)//10, 256))
        self.model = iso.iForest(
            np.array(x_train), ntrees=self.n_trees, sample_size=subsample_size, ExtensionLevel=self.concat_to_channel_size-1)

    @override
    def retraining(self, training_set):
        x_train = []
        for i in range(len(training_set)):
            x_train.extend(self._accumulate_sample(training_set[i]))
        for j in range(self.n_trees):
            if self.performance_counters[j] < 0:
                ix = random.sample(range(self.model.nobjs), self.model.sample)
                X_p = training_set[ix]
                self.model.Trees[j] = iso.iTree(X_p, 0, self.model.limit, exlevel=self.model.exlevel)
            self.performance_counters[j] = 0

    @override
    def predict_current(self):
        self.current_feature_vectors = self.publisher.feature_vectors
        assert len(self.current_feature_vectors.shape) == 3
        n_samples = len(self.current_feature_vectors)
        accumulated_len = len(self._accumulate_sample(self.current_feature_vectors[0]))
        self.current_depths = np.zeros((n_samples, accumulated_len, self.n_trees), dtype=np.float32)
        for k in range(len(self.current_feature_vectors)):
            t1 = time.time()
            x_acc = self._accumulate_sample(self.current_feature_vectors[k])
            t2 = time.time()
            for i in range(accumulated_len):
                for j in range(self.n_trees):
                    self.current_depths[k, i, j] = float(iso.PathFactor(x_acc[i], self.model.Trees[j]).path)     
            print(f'Time for accumulation: {t2 - t1:.6f}s - Time for path calculation: {time.time() - t2:.6f}s')
        self.current_scores = 2.0**(-self.current_depths/self.model.c)
        self.current_predictions = np.mean(self.current_scores, axis=(1, 2))
        
    @override
    def predict(self, x):
        assert len(x.shape) == 3
        result = []
        for i in range(len(x)):
            x_acc = self._accumulate_sample(x[i])
            result.append(2.0**(-np.mean(self.model.compute_paths(x_acc))/self.model.c))
        return result
        
    def update_performance_counters(self, anomaly_score_fn: Callable, anomaly_threshold: float):
        individual_scores = np.mean(self.current_scores, axis=1)
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
            concat_to_channel_size=self.concat_to_channel_size,
            model_id=self.model_id, 
            model_type=self.model_type, 
            debug=self.debug)
        new_instance.model = deepcopy(self.model)
        return new_instance
    
    @override
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
                   
