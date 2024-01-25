import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense


class SimpleRegressionModel(Model):

  def __init__(self):
    super().__init__()
    self.dense1 = Dense(20, activation=tf.nn.relu)
    self.dense2 = Dense(20, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    return x
    