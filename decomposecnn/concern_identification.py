
from time import process_time_ns
from tokenizers import PreTokenizedString
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from utils.tools import extract_weights


def CI_Layers(output,conv_map,first=False):
    flat_output = output.flatten()
        
    if first:
        inactive_nodes = torch.where(flat_output <= 0)[0].tolist()
        conv_map.extend(inactive_nodes)
    else:
        active_nodes = torch.where(flat_output > 0)[0].tolist()
        conv_map = [n for n in conv_map if n not in active_nodes]
        
    return conv_map, output

def CI(model,input,update_weights_dense,update_bias_dense,first=False):
    
    convW, convB, denseW, denseB = extract_weights(model)
    conv_maps = [[] for _ in range(len(convW))]
    layer_info = model.layer_info
    conv_idx = 0
    dense_idx = 0
    output = input
    conv_outputs = []
    dense_outputs=[]
    def hook_fn(module, input, output):
        # print(f"Hook triggered for layer: {module}")
        if isinstance(module, nn.Conv2d):
            conv_outputs.append(output)
        elif isinstance(module, nn.Linear):
            dense_outputs.append(output)
    hooks =[]
    for layer_info in layer_info:
        layer = layer_info['object']
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook) 
    
    with torch.no_grad():
        model(input)
    i=0
    # for layer in layer_info:
    #     output = outputs[i]
    #     print(layer)
    #     i+=1
    #     if layer['type'] == 'conv':
    #         conv_map = conv_maps[conv_idx]
    #         conv_map= CI_Layers(output,conv_map,first)
    #     elif layer['type'] == 'dense':
            
    #         for j in range(len(output)):
    #             # update_weights_dense是更新后的权重 初始化全为0 只要有输入使得当前神经元值大于0 保留所有出边和入边
    #             if output[j] > 0:
    #                 update_weights_dense[dense_idx][j,:] = denseW[dense_idx][j,:]
    #                 update_bias_dense[dense_idx][j] = denseB[dense_idx][j]
    #                 if dense_idx < len(denseW) -2:
    #                     update_weights_dense[dense_idx+1][:j] = denseW[dense_idx+1][:j]
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_map = conv_maps[conv_idx]
            output = conv_outputs[conv_idx]
            conv_map= CI_Layers(output,conv_map,first)
            conv_idx += 1
        elif isinstance(layer, nn.Linear):
            output = dense_outputs[dense_idx][0]
            for j in range(len(output)):
                # update_weights_dense是更新后的权重 初始化全为0 只要有输入使得当前神经元值大于0 保留所有出边和入边
                if output[j] > 0:
                    update_weights_dense[j,:] = denseW[dense_idx][j,:]
                    update_bias_dense[j] = denseB[dense_idx][j]
                    if dense_idx < len(denseW) -2:
                        update_weights_dense[dense_idx+1][:j] = denseW[dense_idx+1][:j]

        

        
    return conv_maps