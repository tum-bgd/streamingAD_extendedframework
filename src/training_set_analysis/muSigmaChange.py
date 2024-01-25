import numpy as np

from abstractTrainingSetAnalysisMethod import AbstractTrainingSetAnalysisMethod
from ..training_set_update.abstractTrainingSetUpdateMethod import AbstractTrainingSetUpdateMethod

class MuSigmaChange(AbstractTrainingSetAnalysisMethod):
    def __init__(self, publisher: AbstractTrainingSetUpdateMethod, models: list, check_every=10) -> None:
        self.publisher = publisher
        self.models = models
        self.reset_all_mu_sigma()
        self.check_every = check_every
        self.counter = 0
        
    def reset_all_mu_sigma(self):
        training_set = self.publisher.get_training_set()
        self.mu0: np.ndarray = np.mean(training_set, axis=0)
        self.var0: np.ndarray = np.var(training_set, axis=0)
        self.running_mu = self.mu0
        self.running_var = self.var0
        self.running_sum = np.sum(training_set, axis=0)
        self.running_sum2 = np.sum(np.square(training_set), axis=0)
        self.counter = 0
    
    def update_parameters(self):
        d = self.publisher.get_last_added_removed()
        N = self.publisher.get_window_length()
        self.running_mu += (1 / N) * (d['last_added'] - d['last_removed'])
        self.running_sum += d['last_added'] - d['last_removed']
        self.running_sum2 += np.square(d['last_added']) - np.square(d['last_removed'])
        self.running_var = (1 / (N-1)) * (self.running_sum2 - (1/N) * np.square(self.running_sum))
        self.counter += 1
        if self.counter == self.check_every:
            self.check_boundaries()
        
    def check_boundaries(self):
        std0 = np.linalg.norm(self.calc_std(self.var0))
        running_std = np.linalg.norm(self.calc_std(self.running_var))
        check_mu: bool = np.linalg.norm(self.running_mu - self.mu0) > std0
        check_std: bool = ((1/2) * std0 > running_std or running_std > 2 * std0)
        if check_mu or check_std:
            self.reset_all_mu_sigma()
            for model in self.models:
                model.retraining(self.publisher.get_training_set())
            
    
    def calc_std(var: np.ndarray):
        return np.sqrt(var)
    
    def notify(self):
        self.update_parameters()