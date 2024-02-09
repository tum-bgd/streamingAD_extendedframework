import numpy as np

from .tsWindowPublisher import TsWindowPublisher
from .abstractSubscriber import AbstractSubscriber

class WindowStreamVectors(AbstractSubscriber):
    def __init__(self, publisher: TsWindowPublisher, window_length: int, subscribers: list[AbstractSubscriber], debug=False) -> None:
        self.publisher:TsWindowPublisher = publisher
        self.window_length = window_length
        self.subscribers = subscribers
        self.debug = debug
    
    def get_feature_vectors(self) -> np.ndarray:
        return self.publisher.window[np.newaxis, -self.window_length:]
    
    def get_last_added(self):
        return np.stack([self.publisher.window[-(self.window_length+i):-i] for i in range(self.publisher.last_update_length, 0, -1)], axis=0)
    
    def add_subscriber(self, new_subscriber):
        self.subscribers.append(new_subscriber)
    
    def add_subscribers(self, new_subscribers: list):
        self.subscribers.extend(new_subscribers)
    
    def notify(self, step_size=1):
        if self.debug:
            from time import time
            t1 = time()
        if self.debug:
            print(f'windowStreamVector at {hex(id(self))} update after {time()-t1:.6f}s')
        if step_size == 1:
            self.feature_vectors = self.get_feature_vectors()
        elif step_size > 1:
            self.feature_vectors = self.get_last_added()
        
        # Iterate over range and not list elements themselves in order to avoid calling newly added models
        for i in range(len(self.subscribers)):
            self.subscribers[i].notify()
        