import numpy as np

from tsWindowPublisher import TsWindowPublisher
from abstractSubscriber import AbstractSubscriber

class WindowStreamVectors(AbstractSubscriber):
    def __init__(self, publisher: TsWindowPublisher, window_length: int, subscribers: list[AbstractSubscriber]) -> None:
        self.publisher:TsWindowPublisher = publisher
        self.window_length = window_length
        self.subscribers = subscribers
    
    def get_feature_vector(self) -> np.ndarray:
        return self.publisher.window[-self.window_length:]
    
    def notify(self):
        for subscriber in self.subscribers:
            subscriber.notify()
        