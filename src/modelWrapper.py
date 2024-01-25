from tensorflow.python.keras import Model

from dataRepresentation import WindowStreamVectors

class ModelWrapper():
    def __init__(self, tfModel: Model, publisher: WindowStreamVectors, subscribers: list) -> None:
        self.tfModel = tfModel
        self.subscribers = subscribers
        self.publisher = publisher
        
    def train(self, x, y, epochs=1):
        self.tfModel.fit(x, y, epochs=epochs)
    
    def predict_current(self):
        self.current_feature_vector = self.publisher.get_feature_vector()
        self.current_feature_vector = self.current_feature_vector.reshape((1, *self.current_feature_vector.shape))
        self.current_prediction = self.tfModel.predict(self.current_feature_vector)
    
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
    
    def notify(self):
        self.predict_current()
        for subscriber in self.subscribers:
            subscriber.notify()