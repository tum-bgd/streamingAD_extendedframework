import math
import numpy as np
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Add, Input, Permute

""" Online ARIMA Tensorflow implementation
OGD version from: 
C. Liu, S. C. Hoi, P. Zhao, and J. Sun, “Online arima algorithms for time
series prediction,” in Proceedings of the AAAI conference on artificial
intelligence, vol. 30, no. 1, 2016.
"""
def get_online_arima(input_shape: "tuple[int]", d: int):
    # 1) Find value N=k+m, based on window length
    window_length = input_shape[0]
    N = window_length - d
    
    # 2) Calculate factors after differencing for 1st term
    diff_factors_2d_first_term = []
    for i in range(1, N):
        diff_factors_2d_first_term.append(calc_diff_factors(i=i, d=d, output_length=window_length))
    diff_factors_first_term = np.stack(diff_factors_2d_first_term, axis=0)
    
    # 3) Differencing factors of 2nd term
    diff_factors_2d_second_term = []
    for j in range(0, d):
        diff_factors_2d_second_term.append(calc_diff_factors(i=1, d=j, output_length=window_length))
    diff_factors_second_term = np.stack(diff_factors_2d_second_term, axis=0)
    
    # 4) Formulate differencing factors as TF layers and combine them in model
    input_shape_transposed = (input_shape[1], input_shape[0])
    inputs = Input(shape=input_shape)
    
    x1 = Permute((2, 1))(inputs)
    x1 = Dense(N-1, use_bias=False, trainable=False, input_shape=input_shape_transposed)(x1)
    x1 = Dense(1, use_bias=False, trainable=True)(x1)
    x1 = Permute((2, 1))(x1)
    first_term = Model(inputs=inputs, outputs=x1)
    first_term.layers[2].set_weights([diff_factors_first_term.T])
    
    x2 = Permute((2, 1))(inputs)
    x2 = Dense(d, use_bias=False, trainable=False, input_shape=input_shape_transposed)(x2)
    x2 = Dense(1, use_bias=False, trainable=False)(x2)
    x2 = Permute((2, 1))(x2)
    second_term = Model(inputs=inputs, outputs=x2)
    second_term.layers[2].set_weights([diff_factors_second_term.T])
    second_term.layers[3].set_weights([np.ones_like(second_term.layers[3].get_weights()[0])])
    
    combined = Add()([x1, x2])
    model = Model(inputs=inputs, outputs=combined)
    return model
        
def calc_diff_factors(i: int, d: int, output_length: int):
    assert i < output_length
    factors = np.zeros((output_length))
    for k in range(0, d+1):
        factors[-(i+k)] = (-1)**k * math.comb(d, k)
    return factors
    