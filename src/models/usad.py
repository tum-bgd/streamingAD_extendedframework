import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input, Permute

def get_usad(input_shape: "tuple[int]", latent_size: int):
    input_shape_transposed = (input_shape[1], input_shape[0])
    in_size = input_shape[0]
    inputs = Input(shape=input_shape)
    x = Permute((2, 1))(inputs)
    
    # Encoder
    enc_dense_1 = Dense(in_size // 2, activation='relu', input_shape=input_shape_transposed)
    enc_dense_2 = Dense(in_size // 4, activation='relu')
    enc_dense_3 = Dense(latent_size, activation='relu')
    x = enc_dense_1(x)
    x = enc_dense_2(x)
    z = enc_dense_3(x)

    # Decoder 1
    y1 = Dense(in_size // 4, activation='relu')(z)
    y1 = Dense(in_size // 2, activation='relu')(y1)
    y1 = Dense(in_size, activation=None)(y1)
    
    # Decoder 2
    dec2_dense_1 = Dense(in_size // 4, activation='relu')
    dec2_dense_2 = Dense(in_size // 2, activation='relu')
    dec2_dense_3 = Dense(in_size, activation=None)
    y2 = dec2_dense_1(z)
    y2 = dec2_dense_2(y2)
    y2 = dec2_dense_3(y2)
    
    # Both decoders
    y3 = enc_dense_1(y1)
    y3 = enc_dense_2(y3)
    y3 = enc_dense_3(y3)
    y3 = dec2_dense_1(y3)
    y3 = dec2_dense_2(y3)
    y3 = dec2_dense_3(y3)

    perm = Permute((2, 1))
    out1, out2, out3 = perm(y1), perm(y2), perm(y3)
    return Model(inputs=inputs, outputs=[out1, out2, out3])