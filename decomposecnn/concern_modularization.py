from turtle import up
import torch
import numpy as np
from utils.tools import extract_weights
import torch.nn.functional as F
delta = 0.5
num_classes = 10
def back_track(model,target_class, UpdatedW, UpdatedB,conv_map,input):
    #改变输出层神经元 num_class -> 2
    convW, convB, denseW, denseB = extract_weights(model)
    last_dense_weight = UpdatedW[-1]
    last_dense_bias = UpdatedB[-1]
    for i in range(len(last_dense_weight)):  # 遍历输出层节点
        temp = last_dense_weight[i,:]  # 当前节点的边权重
        if target_class == 0:
            temp[1] = torch.mean(temp[1:])  # 更新第二节点的权重均值
            temp[2:] = 0
        else:
            # Assign the 1st node as the negative
            tempW, tempB = [], []
            for j in range(num_classes):
                if j != target_class:
                    tempW.append(temp[j])
                    tempB.append(last_dense_bias[j])

            temp[0] = torch.mean(tempW)  # 计算负边的均值
            
            for j in range(num_classes):
                if j != target_class:
                    temp[j] = 0
            
        last_dense_weight[i, :] = temp
    unconcern_node = 0
    if target_class == 0:
        last_dense_bias[1] = torch.mean(last_dense_bias[1:num_classes])  # 更新第二节点的偏置均值
        last_dense_bias[2:num_classes] = 0
        unconcern_nodes =  1
    else:
        last_dense_bias[0] = torch.mean(tempB)
        for j in range(num_classes):
            if j != target_class:
                last_dense_bias[j] = 0
    conv_idx = len(convW)-1

    unconcern_nodes = []
    for i in range(last_dense_weight.shape[1]):
        if last_dense_weight[unconcern_node,i] <= -delta or last_dense_weight[unconcern_node,i] >= +delta:
            unconcern_nodes.append(j)
    
    layer_info = model.layer_info
    for i in range(len(layer_info) - 2, 0, -1):
        layer = layer_info[i]
        if layer['type'] == 'conv':
            conv_map[conv_idx] = CMBI(input,convW[conv_idx],layer_info[i-1]['type'],conv_map[conv_idx])
            conv_idx-=1
        elif layer['params']['weight'] is not None:
            nodes_to_remove = []
            
            for unconcern_node in unconcern_nodes:
                if UpdatedW[i][unconcern_node]<= -delta or UpdatedW[i][unconcern_node] >= +delta:
                    nodes_to_remove.append(j)
            for node in nodes_to_remove:
                UpdatedW[i][node] = 0
                UpdatedB[i][node] = 0
            unconcern_nodes = nodes_to_remove
            

            # if len(temp) > 0:
            #     UpdatedW[-1] = np.delete(UpdatedW[-1], temp, axis=0)
            #     UpdatedB[-1] = np.delete(UpdatedB[-1], temp, axis=0)
    return UpdatedW, UpdatedB, conv_map

def Sliding_Window_Mapping(input, W, pad, stride):
    """
    Perform the forward pass and map input-output nodes based on sliding window operation.
    :param input: Input tensor.
    :param W: Convolutional kernel weights.
    :param pad: Padding applied to input tensor.
    :param stride: Stride used in the convolution operation.
    :return: Mapping of input nodes to output nodes.
    """
    mapping = []
    temp = torch.zeros_like(input)  # Initialize temporary tensor with same size as input
    temp = temp.flatten()

    # Add indices for all input nodes
    for i in range(temp.shape[0]):
        temp[i] = i + 1

    temp = temp.flatten()  # Flatten for sliding window operation
    sliding_window = F.unfold(input, kernel_size=W.shape[2:], padding=pad, stride=stride)  # Apply convolution to get unfolded output

    for i in range(sliding_window.shape[1]):  # Iterate over sliding window
        for j in range(W.shape[3]):  # Iterate through the filter depth
            mapping.append((temp[i + j * sliding_window.shape[1]].item(), i))  # Add input-output mapping

    return mapping

def CMBI(input, W, preceding_layer, deactive_map, pad=1, stride=1):
    """
    Backtrack to the input layer by performing a source-sink mapping and updating the deactivation map.
    :param input: Input tensor.
    :param W: Convolutional kernel weights.
    :param pad: Padding applied to input tensor.
    :param stride: Stride used in the convolution operation.
    :param B: Bias term.
    :param preceding_layer: Previous layer in the network, used for "Add" layer checking.
    :param deactive_map: Map of deactivated nodes.
    :return: Updated deactivation map after backtracking.
    """
    convDepth = len(W)  # Depth of convolutional layers
    mapping_window = Sliding_Window_Mapping(input, W, pad, stride)  # Get the mapping from input to output nodes

    source_mapping = mapping_window[:0]  # Input nodes
    sink_mapping = mapping_window[1:]   # Output nodes

    # Iterate through all deactivated nodes and perform source-sink mapping
    for deactivated_node in deactive_map:
        source = source_mapping[sink_mapping == deactivated_node]
        flag = True
        source_mapping = source_mapping[source_mapping > 0]

        for source_node in source_mapping:
            if flag:
                sink_node = sink_mapping[source_mapping == source_node]
                if abs(sink_node - deactive_map) > 0:
                    flag = False

        if flag:
            for source in source_mapping:
                if source not in deactive_map:
                    deactive_map[-1].add(source + 1)  # Update map

            if preceding_layer == "Add":
                deactive_map[-2].add(source + 1)  # Update map for "Add" layer

    return deactive_map

