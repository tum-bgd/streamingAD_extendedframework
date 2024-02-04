import tensorflow as tf
from copy import copy
from tensorflow.python.keras.models import clone_model

from .dataRepresentation import WindowStreamVectors


class ModelWrapper():
    def __init__(self, tf_model: tf.keras.Model, publisher: WindowStreamVectors, subscribers: list, model_id: str,
                 model_type: str, debug=False) -> None:
        self.tf_model = tf_model
        self.subscribers = subscribers
        self.publisher = publisher
        self.model_id = model_id
        self.model_type = model_type
        self.batch_size = 32
        self.debug = debug

        if self.model_type == 'reconstruction':
            # self.tf_model.build(input_shape=input_shape)
            self.tf_model.compile(loss=tf.keras.losses.mae, optimizer="adam")
        if self.model_type == 'forecasting':
            # self.tf_model.build(input_shape=(input_shape[0]-1, input_shape[1]))
            self.tf_model.compile(loss=tf.keras.losses.mae, optimizer="adam")

    def train(self, x, epochs):
        if self.model_type == 'reconstruction':
            self.tf_model.fit(x, x, batch_size=self.batch_size, epochs=epochs)
        if self.model_type == 'forecasting':
            self.tf_model.fit(x[:, :-1], x[:, -1:],
                              batch_size=self.batch_size, epochs=epochs)

    def predict_current(self):
        self.current_feature_vector = self.publisher.get_feature_vector()
        self.current_feature_vector = self.current_feature_vector.reshape(
            (1, *self.current_feature_vector.shape))
        self.current_prediction = self.tf_model.predict(
            self.current_feature_vector)

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def retraining(self, training_set):
        self.train(training_set, 1)

    def factory_copy(self):
        return ModelWrapper(
            tf_model=clone_model(self.tf_model),
            publisher=self.publisher,
            subscribers=copy(self.subscribers),
            model_id=self.model_id,
            model_type=self.model_type,
            debug=self.debug)

    def notify(self):
        if self.debug:
            from time import time
            t1 = time()
        self.predict_current()
        if self.debug:
            print(f'{self.model_id} at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify()
