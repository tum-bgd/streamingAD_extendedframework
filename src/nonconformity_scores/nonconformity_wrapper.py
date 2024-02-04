import os
import numpy as np
import pandas as pd
from copy import copy

from ..abstractSubscriber import AbstractSubscriber
from ..modelWrapper import ModelWrapper


class NonConformityWrapper(AbstractSubscriber):
    def __init__(self, publisher: ModelWrapper, subscribers: list, save_paths: list[str], debug=False) -> None:
        self.subscribers = subscribers
        self.publisher = publisher
        self.save_paths = save_paths
        self.debug = debug

    def calc_current_nonconformity_score(self):
        self.current_feature_vector = self.publisher.current_feature_vector
        self.nonconformity_score = calc_nonconformity_scores(
            self.publisher.current_feature_vector, self.publisher.current_prediction)

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def add_subscribers(self, subscribers: list):
        self.subscribers.extend(subscribers)
        
    def get_update_index(self):
        return self.publisher.publisher.publisher.get_update_index()

    def save_nonconformity_score(self):
        index = self.get_update_index()
        for save_path in self.save_paths:
            parent_dir = '/'.join(save_path.split('/')[:-1])
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            entry = pd.DataFrame([self.nonconformity_score], columns=[
                'nonconformity_score'], index=[index])
            if not os.path.exists(save_path):
                entry.index.name = 'counter'
                entry.to_csv(save_path, mode='w')
            else:
                entry.to_csv(save_path, mode='a', header=False)

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        self.calc_current_nonconformity_score()
        self.save_nonconformity_score()
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
            debug=self.debug)
        new_instance.nonconformity_score = self.nonconformity_score
        return new_instance


def calc_nonconformity_scores(feature_vectors: np.ndarray, predictions: np.ndarray, measure='euclidean'):
    if measure == 'euclidean':
        axis = (1, 2) if len(feature_vectors.shape) == 3 else None
        return np.linalg.norm(feature_vectors - predictions, axis=axis)
    elif measure == 'mean_abs_diff':
        return np.mean(np.abs(feature_vectors - predictions))
    elif measure == 'cosine_sim':
        return np.sum(feature_vectors * predictions) / (np.linalg.norm(feature_vectors) * np.linalg.norm(predictions))
    elif measure == 'pearson':
        return (1/(feature_vectors.shape[1])) * \
            np.sum((feature_vectors - np.mean(feature_vectors, axis=1, keepdims=True)) *
                   (predictions - np.mean(predictions, axis=1, keepdims=True))) / \
            (np.std(feature_vectors) * np.std(predictions))
