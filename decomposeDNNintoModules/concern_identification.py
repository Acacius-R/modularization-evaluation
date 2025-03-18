import numpy as np
from utils import *
def concern_identification(model, input_data, indicator, D, b):
    """
    CI alorighthm
    """
    
    weights, biases = get_weights_and_biases(model)
 

    
    layer_outputs = get_layer_outputs(input_data, weights, biases)
    layer_num = len(weights)
    # print(layer_num)
    
    for l in range(layer_num):
        for j in range(len(layer_outputs[l])):
            
            if layer_outputs[l][j] <=0: 
                
                D[l][j, :] = 0  
                
                if l < layer_num-2:
                    D[l + 1][:, j] = 0
                b[l][j] = 0  
            else:
                if indicator:
                    D[l][:, j] = weights[l][:, j]
                    b[l][j] = biases[l][j]
                else:
                    for k in range(weights[l].shape[1]):  # 遍历边
                        if weights[l][j, k] < 0:
                            
                            D[l][j, k] = max(D[l][j, k], weights[l][j, k])
                            if D[l][j, k] < 0:
                                D[l][j, k] = 0
                        else:
                            D[l][j, k] = min(D[l][j, k], weights[l][j, k])
                    b[l][j] = biases[l][j]
    
    
    output_size = len(layer_outputs[-1])
    for j in range(output_size):
        if indicator:
            D[-1][:, j] = weights[-1][:, j]
            b[-1][j] = biases[-1][j]
        else:
            for k in range(weights[-1].shape[1]):
                if weights[-1][j, k] < 0:
                    D[-1][j, k] = max(D[-1][j, k], weights[-1][j, k])
                else:
                    D[-1][j, k] = min(D[-1][j, k], weights[-1][j, k])
            b[-1][j] = biases[-1][j]
        
    return D, b
