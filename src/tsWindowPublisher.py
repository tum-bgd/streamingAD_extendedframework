import numpy as np


class TsWindowPublisher:    
    def __init__(self, dataset: np.ndarray, window_length: int, subscribers: list, debug=False) -> None:
        self.dataset = dataset
        self.update_index = 0
        self.window = dataset[:window_length]
        self.window_length = window_length
        self.subscribers = subscribers
        self.last_update_length = window_length
        self.last_added = dataset[:window_length]
        self.last_removed = np.zeros(())
        self.debug = debug
        
    def update_window(self):
        if self.debug:
            from time import time
            t1 = time()
        self.update_index += 1
        self.last_removed = self.window[:1]
        self.window = self.dataset[self.update_index : self.update_index + self.window_length]
        self.last_added = self.window[-1:]
        self.last_update_length = 1
        if self.debug:
            print(f'tsWindowPublisher at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()
        return 0
    
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0
        
    def remove_subscriber(self, subscriber):
        self.subscribers.remove(subscriber)
        return 0
    
    def get_update_index(self):
        return self.update_index
    
    def get_window(self):
        return self.window
    
    def get_window_length(self):
        return self.window_length