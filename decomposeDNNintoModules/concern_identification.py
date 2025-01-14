import numpy as np
from utils import *
def concern_identification(model, input_data, indicator, D, b):
    """
    实现 Concern Identification (CI) 算法。
    
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
    layer_num = len(weights)
    # print(layer_num)
    #逐层更新 D 和 b
    for l in range(layer_num):
        for j in range(len(layer_outputs[l])):
            #遍历每一层l和每一层的节点j
            if layer_outputs[l][j] <=0: 
                # 不活跃节点，则第l层节点j到下一层所有节点的边都置为0
                D[l][j, :] = 0  # 更新边矩阵
                # 将下一层中所有连接到当前节点（j）的边置为 0？这里为什么这样写
                if l < layer_num-2:
                    D[l + 1][:, j] = 0
                b[l][j] = 0  # 更新偏置向量
            else:
                if indicator:
                    D[l][:, j] = weights[l][:, j]
                    b[l][j] = biases[l][j]
                else:
                    for k in range(weights[l].shape[1]):  # 遍历边
                        if weights[l][j, k] < 0:
                            # D和weights有什么区别？
                            D[l][j, k] = max(D[l][j, k], weights[l][j, k])
                            if D[l][j, k] < 0:
                                D[l][j, k] = 0
                        else:
                            D[l][j, k] = min(D[l][j, k], weights[l][j, k])
                    b[l][j] = biases[l][j]
    
    #更新输出层边和偏置
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
