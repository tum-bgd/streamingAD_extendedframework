from nbeats_keras.model import NBeatsNet

def get_nbeats(input_shape: tuple[int]):
    return NBeatsNet(
        input_dim=input_shape[1], 
        output_dim=input_shape[1],
        backcast_length=input_shape[0],
        forecast_length=1,
        stack_types=(NBeatsNet.GENERIC_BLOCK,),
        nb_blocks_per_stack=3,
        thetas_dim=(4,),
        share_weights_in_stack=True,
        hidden_layer_units=50,
    )
