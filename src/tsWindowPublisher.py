import numpy as np


class TsWindowPublisher:    
    def __init__(self, dataset: np.ndarray, subscribers: list) -> None:
        self.window = first_window
        self.window_length = len(first_window)
        self.subscribers = subscribers
        self.last_update_length = self.window_length
        self.last_added = first_window
        self.last_removed = np.zeros_like(first_window[0:1])
        
    def update_window(self, new_window_values: np.ndarray):
        self.last_added = new_window_values
        self.last_removed = self.window[:len(new_window_values)]
        self.window = np.concatenate([self.window[len(new_window_values):], new_window_values])
        self.last_update_length = len(new_window_values)
        for subscriber in self.subscribers:
            subscriber.notify()
        return 0
    
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0
        
    def remove_subscriber(self, subscriber):
        self.subscribers.remove(subscriber)
        return 0
        
    def get_window_length(self):
        return self.window_length