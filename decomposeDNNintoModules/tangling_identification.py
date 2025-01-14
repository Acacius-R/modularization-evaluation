import numpy as np
from utils import *

def tangling_identification(model, input_data, indicator, D, b):
    """
    实现 Tangling Identification (CI) 算法。
    
    参数:
        model: 已训练的模型 (含权重和偏置)。
        input_data: 输入数据 (X)。
        indicator: 布尔值，指示是否更新权重。
        D: 边的权重矩阵（初始为零矩阵）。
        b: 偏置向量（初始为零向量）。
    
    返回:
        更新后的边矩阵 D 和偏置向量 b。
    """
    #获取权重和偏置
    weights, biases = get_weights_and_biases(model)
    
    #获取每层的输出
    layer_outputs = get_layer_outputs(input_data, weights, biases)

    for l in range(len(layer_outputs)-1):
        for j in range(len(layer_outputs[l])):
            if layer_outputs[l][j] <= 0:
                D[l][:, j] = [x for x in D[l][:, j]]

            else:
                #活跃节点
                D[l][:, j] = weights[l][:, j]
                b[l][j] = biases[l][j]

    
    #更新输出层边和偏置
    output_size = len(layer_outputs[-1])
    for j in range(output_size):
        if layer_outputs[-1][j] > 0.0001:
            D[-1][:, j] = weights[-1][:, j]
            b[-1][j] = biases[-1][j]
        
    return D, b
