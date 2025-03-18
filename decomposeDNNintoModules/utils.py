import numpy as np

def softmax(x):
    """

    """

    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    # subtract the maximum value from the input vector 
    # not change the result of the softmax, beacause the relative scale of exp remains unchanged.

#return the outputs of every layers
def get_layer_outputs(input_data,weights,biases):
    x = input_data
    # x.reshape(x.shape[0], 28*28)


    layer_outputs = []
    for i in range(len(weights)-1):
        x = np.dot(x, weights[i])
        x = np.add(x, biases[i])
        layer_outputs.append(x)


    x = np.dot(x, weights[-1])
    x = np.add(x, biases[-1])
    x = softmax(x)
    layer_outputs.append(x)

    return layer_outputs

# must be Keras or TensorFlow
def get_weights_and_biases(model):
    """
    获取模型的权重和偏置。
    """
    # skip flatten layer
    weights = [layer.get_weights()[0] for layer in model.layers[1:]]
    biases = [layer.get_weights()[1] for layer in model.layers[1:]]
    return weights, biases
