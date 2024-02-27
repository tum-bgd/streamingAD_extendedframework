import os
import numpy as np
import pandas as pd
from copy import copy

from abstractSubscriber import AbstractSubscriber
from modelWrapper import ModelWrapper


class NonConformityWrapper(AbstractSubscriber):
    def __init__(self, publisher: ModelWrapper, subscribers: list, save_paths: "list[str]", measure='euclidean', debug=False) -> None:
        self.subscribers = subscribers
        self.publisher = publisher
        self.save_paths = save_paths
        self.debug = debug
        self.measure = measure
        self.max_value = 1

    def calc_current_nonconformity_scores(self):
        self.current_feature_vectors = self.publisher.current_feature_vectors
        if self.publisher.model_type in ['forecasting', 'reconstruction']:
            assert len(self.current_feature_vectors.shape) == 3
            assert len(self.publisher.current_predictions.shape) == 3
            if self.publisher.model_type == 'forecasting':
                predictions = self.publisher.current_predictions[:, -1:]
                to_predict = self.publisher.current_feature_vectors[:, -1:]
                if self.current_feature_vectors.shape[2] == 1:
                    measure = 'mean_abs_diff'
                else:
                    measure = self.measure  
                self.nonconformity_scores = calc_nonconformity_scores(
                    to_predict, predictions, measure)
                if measure in ['euclidean', 'mean_abs_diff']:
                    max_ns = np.max(self.nonconformity_scores)
                    if max_ns > self.max_value:
                        self.max_value = max_ns
                    self.nonconformity_scores /= self.max_value
                    # self.nonconformity_scores /= max_ns
            elif self.publisher.model_type == 'reconstruction':
                predictions = self.publisher.current_predictions
                to_predict = self.publisher.current_feature_vectors
                self.nonconformity_scores = calc_nonconformity_scores(
                    to_predict, predictions, measure=self.measure)
                    
        elif self.publisher.model_type == 'iforest':
            self.nonconformity_scores = self.publisher.current_predictions

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def add_subscribers(self, subscribers: list):
        self.subscribers.extend(subscribers)

    def get_update_index(self):
        return self.publisher.publisher.publisher.get_update_index()

    def save_nonconformity_scores(self):
        update_index = self.get_update_index()
        for save_path in self.save_paths:
            parent_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            entry = pd.DataFrame(self.nonconformity_scores, columns=[
                'nonconformity_score'], index=[i for i in range(update_index-len(self.nonconformity_scores)+1, update_index+1)])
            if not os.path.exists(save_path):
                entry.index.name = 'counter'
                entry.to_csv(save_path, mode='w')
            else:
                entry.to_csv(save_path, mode='a', header=False)

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        self.calc_current_nonconformity_scores()
        self.save_nonconformity_scores()
        if self.debug:
            print(
                f'NonconformityWrapper at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()

    def factory_copy(self):
        new_instance = NonConformityWrapper(
            publisher=self.publisher,
            subscribers=copy(self.subscribers),
            save_paths=copy(self.save_paths),
            measure=self.measure,
            debug=self.debug)
        new_instance.nonconformity_scores = self.nonconformity_scores
        new_instance.max_value = self.max_value
        return new_instance


def calc_nonconformity_scores(feature_vectors: np.ndarray, predictions: np.ndarray, measure: str):
    axis = (1, 2) if len(feature_vectors.shape) == 3 else \
        (2, 3) if len(feature_vectors.shape) == 4 else None
    if measure == 'euclidean':
        return np.linalg.norm(feature_vectors - predictions, axis=axis)
    elif measure == 'mean_abs_diff':
        return np.mean(np.abs(feature_vectors - predictions), axis=axis)
    elif measure == 'cosine_sim':
        return 1 - np.sum(feature_vectors * predictions, axis=axis) / (np.linalg.norm(feature_vectors, axis=axis) * np.linalg.norm(predictions, axis=axis))
    # elif measure == 'pearson':
    #     return (1/(feature_vectors.shape[1])) * \
    #         np.sum((feature_vectors - np.mean(feature_vectors, axis=1, keepdims=True)) *
    #                (predictions - np.mean(predictions, axis=1, keepdims=True))) / \
    #         (np.std(feature_vectors) * np.std(predictions))
    elif measure == 'iforest':
        return predictions
