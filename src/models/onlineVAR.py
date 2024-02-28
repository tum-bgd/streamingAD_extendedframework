import time
import numpy as np
from statsmodels.tsa.api import VAR
import random
from typing_extensions import override
from copy import deepcopy
import pickle
from typing import Callable

from dataRepresentation import WindowStreamVectors
from modelWrapper import ModelWrapper


class OnlineVAR(ModelWrapper):
    def __init__(self, publisher: WindowStreamVectors, subscribers: list, model_id: str,
                 model_type: str, lag_order=None, debug=False) -> None:
        self.subscribers = subscribers
        self.publisher = publisher
        self.model_id = model_id
        self.model_type = model_type
        self.model = None
        self.dataset_train = None
        self.lag_order = lag_order
        self.debug = debug

    @override
    def train(self, x: np.ndarray, epochs):
        assert len(x.shape) == 2
        self.dataset_train = x
        window_length, n_channels = x.shape
        assert n_channels > 1
        model = VAR(x)
        self.model = model.fit(maxlags=50, ic='aic') if self.lag_order is None else model.fit(self.lag_order)
        # self.model = model.fit(self.lag_order)
        if self.model.k_ar != 0:
            self.lag_order = self.model.k_ar
        else:
            print('Automatic lag order selection failed!')
            raise Exception
        print(f'Trained VAR model with automatic lag order selection')
        print(self.model.summary())

    @override
    def retraining(self, training_set):
        assert len(training_set.shape) == 3
        # assuming consecutive instances in training set
        self.train(training_set[:, 0, :], 1)

    @override
    def predict_current(self):
        self.current_feature_vectors = self.publisher.feature_vectors
        cfv_shape = self.current_feature_vectors.shape
        assert len(cfv_shape) == 3
        self.current_predictions = np.zeros((cfv_shape[0], 1, cfv_shape[2]))
        for i in range(cfv_shape[0]):
            self.current_predictions[i, 0] = self.model \
                .forecast(self.current_feature_vectors[i, -(self.lag_order+1):-1], 1)
    
    @override
    def predict(self, x):
        assert len(x.shape) == 3
        predictions = np.zeros((x.shape[0], x.shape[2]))
        for i in range(len(x)):
            predictions[i] = self.model.forecast(x[i, -(self.lag_order+1):-1], 1)
        return predictions
    
    @override
    def factory_copy(self):
        new_instance = OnlineVAR(
            publisher=self.publisher, 
            subscribers=self.subscribers.copy(),
            lag_order=self.lag_order,
            model_id=self.model_id, 
            model_type=self.model_type, 
            debug=self.debug)
        new_instance.model = deepcopy(self.model)
        return new_instance
    
    @override
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
                   
