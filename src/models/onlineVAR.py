import time
import numpy as np
from statsmodels.tsa.api import VAR
import random
from typing_extensions import override
from copy import deepcopy
import pickle
from typing import Callable

from ..dataRepresentation import WindowStreamVectors
from ..modelWrapper import ModelWrapper


class OnlineVAR(ModelWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(tf_model=None, *args, **kwargs)
        self.model = None
        self.dataset_train = None

    @override
    def train(self, dataset_train: np.ndarray, epochs):
        assert len(dataset_train.shape) == 2
        self.dataset_train = dataset_train
        window_length, n_channels = dataset_train.shape
        assert n_channels > 1
        model = VAR(dataset_train)
        self.model = model.fit(maxlags=50, ic='aic')
        print(f'Trained VAR model with automatic lag order selection')
        print(self.model.summary())
        
    @override
    def retraining(self, training_set):
        pass

    @override
    def predict_current(self):
        pass
    
    @override
    def predict(self, x):
        pass
    
    @override
    def factory_copy(self):
        new_instance = OnlineVAR(
            publisher=self.publisher, 
            subscribers=self.subscribers.copy(),
            model_id=self.model_id, 
            model_type=self.model_type, 
            debug=self.debug)
        new_instance.model = deepcopy(self.model)
        return new_instance
    
    @override
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
                   
