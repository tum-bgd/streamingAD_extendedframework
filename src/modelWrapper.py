import numpy as np
import tensorflow as tf
from copy import copy
from tensorflow.python.keras.models import clone_model

from dataRepresentation import WindowStreamVectors
# from models.nbeats import get_nbeats


class ModelWrapper():
    def __init__(self, tf_model: tf.keras.Model, publisher: WindowStreamVectors, subscribers: list, model_id: str,
                 model_type: str, debug=False) -> None:
        self.tf_model = tf_model
        self.subscribers = subscribers
        self.publisher = publisher
        self.model_id = model_id
        self.model_type = model_type
        self.batch_size = 32
        self.output_index = None
        self.epoch = 1
        self.debug = debug

        if self.model_type == 'reconstruction':
            # self.tf_model.build(input_shape=input_shape)
            if self.model_id == 'usad':
                self.tf_model.compile(loss=[tf.keras.losses.mse, tf.keras.losses.mse, tf.keras.losses.mse], loss_weights=[1/self.epoch, 1/self.epoch, 1 - 1/self.epoch], optimizer="adam")
            else:
                self.tf_model.compile(loss=tf.keras.losses.mae, optimizer="adam")
        if self.model_type == 'forecasting':
            # self.tf_model.build(input_shape=(input_shape[0]-1, input_shape[1]))
            self.tf_model.compile(loss=tf.keras.losses.mae, optimizer="adam")

    def _set_output_index(self, index):
        self.output_index = index

    def train(self, x, epochs):
        x_tf = tf.convert_to_tensor(x)
        if self.model_type == 'reconstruction':
            for i in range(epochs):
                history = self.tf_model.fit(
                    x_tf, x_tf, batch_size=self.batch_size, epochs=1)
                self.epoch += 1
            if self.model_id == 'usad':
                self._set_output_index(
                    np.argmin(np.array(list(history.history.values())[1:])[:, -1]))
        if self.model_type == 'forecasting':
            for i in range(epochs):
                self.tf_model.fit(x_tf[:, :-1], x_tf[:, -1:],
                                  batch_size=self.batch_size, epochs=1)
                self.epoch += 1

    def predict_current(self):
        self.current_feature_vectors = self.publisher.feature_vectors
        if self.model_type == 'reconstruction':
            self.current_predictions = self.tf_model.predict(
                tf.convert_to_tensor(self.current_feature_vectors), verbose=0)
        if self.model_type == 'forecasting':
            self.current_predictions = self.tf_model.predict(
                tf.convert_to_tensor(self.current_feature_vectors[:, :-1]), verbose=0)
        if self.output_index is not None:
            self.current_predictions = self.current_predictions[self.output_index]

    def predict(self, x):
        assert len(x.shape) == 3
        if self.model_type == 'reconstruction':
            preds = self.tf_model.predict(x=x)
        if self.model_type == 'forecasting':
            preds = self.tf_model.predict(x=x[:, :-1])

        if self.output_index is not None:
            return preds[self.output_index]
        else:
            return preds

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def retraining(self, training_set):
        self.train(training_set, 5)

    def save_model(self, save_path):
        self.tf_model.save_weights(save_path)

    def factory_copy(self):
        if self.model_id != 'nbeats':
            model_clone = clone_model(self.tf_model)
        else:
            pass
            # model_clone = get_nbeats(
            #     input_shape=self.tf_model.input_shape)
            # model_clone.set_weights(self.tf_model.get_weights())
        return ModelWrapper(
            tf_model=model_clone,
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
