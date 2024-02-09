import os
import numpy as np
import pandas as pd
from copy import copy

from ..tsWindowPublisher import TsWindowPublisher
from .abstractAnomalyScore import AbstractAnomalyScore
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod
from ..nonconformity_scores.nonconformity_wrapper import NonConformityWrapper


class AverageOfWindow(AbstractAnomalyScore):
    def __init__(self, publisher: NonConformityWrapper, ts_window_publisher: TsWindowPublisher, subscribers: list,
                 save_paths: "list[str]", initial_nonconformity_scores: np.ndarray, window_length: int, debug=False) -> None:
        self.publisher = publisher
        self.ts_window_publisher = ts_window_publisher
        self.subscribers = subscribers
        self.window_length = window_length
        self.save_paths = save_paths
        self.window = initial_nonconformity_scores[-window_length:]
        self.running_mean = np.mean(
            initial_nonconformity_scores[-window_length:])
        self.debug = debug

    def update_parameters(self):
        to_add = self.publisher.nonconformity_scores
        step_size = len(to_add)
        to_remove = self.window[:step_size]
        self.anomaly_scores = np.array(
            [self.running_mean + (1/self.window_length) * (np.sum(to_add[:i]) - np.sum(to_remove[:i])) for i in range(1, step_size+1)])
        self.window = np.concatenate([self.window[step_size:], to_add])
        self.running_mean += (1/self.window_length) * \
            (np.sum(to_add) - np.sum(to_remove))

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
        self.update_parameters()
        self.save_anomaly_scores()
        if self.debug:
            print(
                f'AvgOfWindow at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()

    def get_anomaly_scores(self) -> np.ndarray:
        return self.anomaly_scores

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def factory_copy(self):
        new_instance = AverageOfWindow(
            publisher=self.publisher,
            ts_window_publisher=self.ts_window_publisher,
            subscribers=copy(self.subscribers),
            save_paths=copy(self.save_paths),
            initial_nonconformity_scores=self.window.copy(),
            window_length=self.window_length,
            debug=self.debug)
        new_instance.anomaly_scores = self.anomaly_scores
        return new_instance
