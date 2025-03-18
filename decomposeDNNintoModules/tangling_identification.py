import numpy as np
from utils import *

def tangling_identification(model, input_data, indicator, D, b):
    """
    TI algorithm
    """

    weights, biases = get_weights_and_biases(model)
    

    layer_outputs = get_layer_outputs(input_data, weights, biases)

    for l in range(len(layer_outputs)-1):
        for j in range(len(layer_outputs[l])):
            if layer_outputs[l][j] <= 0:
                D[l][:, j] = [x for x in D[l][:, j]]

            else:

                D[l][:, j] = weights[l][:, j]
                b[l][j] = biases[l][j]

    

    output_size = len(layer_outputs[-1])
    for j in range(output_size):
        if layer_outputs[-1][j] > 0.0001:
            D[-1][:, j] = weights[-1][:, j]
            b[-1][j] = biases[-1][j]
        
    return D, b
