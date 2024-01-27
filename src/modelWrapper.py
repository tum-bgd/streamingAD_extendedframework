from tensorflow.python.keras import Model

from dataRepresentation import WindowStreamVectors

class ModelWrapper():
    def __init__(self, tfModel: Model, publisher: WindowStreamVectors, subscribers: list, model_id: str, model_type: str) -> None:
        self.tfModel = tfModel
        self.subscribers = subscribers
        self.publisher = publisher
        self.model_id = model_id
        self.model_type = model_type
        self.batch_size = 32
        
    def train(self, x, epochs):
        if self.model_type == 'reconstruction':
            self.tfModel.fit(x, x, batch_size=self.batch_size, epochs=epochs)
        if self.model_type == 'forecasting':
            self.tfModel.fit(x[:, :-1], x[:, -1:], batch_size=self.batch_size, epochs=epochs)
        else:
            pass
    
    def predict_current(self):
        self.current_feature_vector = self.publisher.get_feature_vector()
        self.current_feature_vector = self.current_feature_vector.reshape((1, *self.current_feature_vector.shape))
        self.current_prediction = self.tfModel.predict(self.current_feature_vector)
    
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
    
    def retraining(self, training_set):
        self.train(training_set, 1)
    
    def notify(self):
        self.predict_current()
        for subscriber in self.subscribers:
            subscriber.notify()