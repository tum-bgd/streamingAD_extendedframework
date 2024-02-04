import numpy as np

from .tsWindowPublisher import TsWindowPublisher
from .abstractSubscriber import AbstractSubscriber

class WindowStreamVectors(AbstractSubscriber):
    def __init__(self, publisher: TsWindowPublisher, window_length: int, subscribers: list[AbstractSubscriber], debug=False) -> None:
        self.publisher:TsWindowPublisher = publisher
        self.window_length = window_length
        self.subscribers = subscribers
        self.debug = debug
    
    def get_feature_vector(self) -> np.ndarray:
        return self.publisher.window[-self.window_length:]
    
    def add_subscriber(self, new_subscriber):
        self.subscribers.append(new_subscriber)
    
    def add_subscribers(self, new_subscribers: list):
        self.subscribers.extend(new_subscribers)
    
    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        if self.debug:
            print(f'windowStreamVector at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()
        