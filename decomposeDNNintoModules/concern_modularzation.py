import numpy as np
num_classes = 10
def cm_c(D,b,target_class):
    """
    实现 Concern Modularization: Channeling (CM-C) 算法。

    参数:
        D: 输出层的边矩阵。
        b: 偏置向量。
        target_class: 目标类的标签。

    返回:
        更新后的 D 和 b。
    """
    last_layer_weights = D[-1]
    last_layer_biases = b[-1]
    for i in range(len(last_layer_weights)):  # 遍历输出层节点
        temp = last_layer_weights[i,:]  # 当前节点的边权重
        if target_class == 0:
            temp[1] = np.mean(temp[1, :])  # 更新第二节点的权重均值
            temp[2, :] = 0
        else:
            # Assign the 1st node as the negative
            tempW, tempB = [], []
            for j in range(len(num_classes)):
                if j != target_class:
                    tempW.append(temp[j])
                    tempB.append(last_layer_biases[j])

            temp[0] = np.mean(tempW)  # 计算负边的均值
            
            for j in range(len(num_classes)):
                if j != target_class:
                    temp[j] = 0
            
        last_layer_weights[i, :] = temp

    if target_class == 0:
        last_layer_biases[1] = np.mean(last_layer_biases[1:num_classes])  # 更新第二节点的偏置均值
        last_layer_biases[2:num_classes] = 0
    else:
        last_layer_biases[0] = np.mean(tempB)
        for j in range(len(num_classes)):
            if j != target_class:
                last_layer_biases[j] = 0

    #移除不相关的边
    irrelevant_nodes = []
    for i in range(last_layer_weights.shape[0]):
        if target_class ==0:
            if last_layer_weights[i,0] == 0 and last_layer_weights[i,1] != 0:
                irrelevant_nodes.append(i)
        else:
            if last_layer_weights[i,0] != 0 and last_layer_weights[i,target_class] == 0:
                irrelevant_nodes.append(i)

    if (len(irrelevant_nodes) > 0):
        if target_class ==0:
            tempD=[]
            tempB=[]
            for node in irrelevant_nodes:
                tempD.append(last_layer_weights[node,1])
                tempB.append(last_layer_biases[node])

                last_layer_weights[node,1] =0
                last_layer_biases[node] = 0

            last_layer_weights[irrelevant_nodes[0],1] = np.mean(tempD)
            last_layer_biases[irrelevant_nodes[0]] = np.mean(tempB)
        else:
            tempD=[]
            tempB=[]
            for node in irrelevant_nodes:
                tempD.append(last_layer_weights[node,0])#源代码错误？：tempD2.append(self.D2[x,1])
                tempB.append(last_layer_biases[node])

                last_layer_weights[node,0] =0
                last_layer_biases[node] = 0

            last_layer_weights[irrelevant_nodes[0],0] = np.mean(tempD)
            last_layer_biases[irrelevant_nodes[0]] = np.mean(tempB)

    

    return D, b