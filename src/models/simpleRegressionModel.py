import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Reshape, InputLayer


class SimpleRegressionModel(Model):
  def __init__(self, input_shape: tuple[int]):
    super().__init__()
    self.reshape1 = Reshape((input_shape[0] * input_shape[1],), input_shape=input_shape)
    self.dense1 = Dense(20, activation=tf.nn.relu)
    self.dense2 = Dense(input_shape[0] * input_shape[1])
    self.reshape2 = Reshape((input_shape[0], input_shape[1]))

  def call(self, inputs):
    x = self.reshape1(inputs)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.reshape2(x)
    return x
  
def get_simple_regression_model(input_shape: tuple[int]):
  return Sequential([
    InputLayer(input_shape=input_shape),
    Reshape((input_shape[0] * input_shape[1],), input_shape=input_shape),
    Dense(20, activation=tf.nn.relu),
    Dense(input_shape[0] * input_shape[1]),
    Reshape((input_shape[0], input_shape[1]))
  ])
    