from abc import ABC, abstractmethod


class AbstractSubscriber(ABC):
    @abstractmethod
    def notify():
        pass

