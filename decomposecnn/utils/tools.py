

def extract_weights(model):
    convW, convB, denseW, denseB = [], [], [], []
    for layer in model.layer_info:
        if layer['type'] == 'conv':
            convW.append(layer['params']['weight'])
            convB.append(layer['params']['bias'])
        elif layer['type'] == 'dense':
            denseW.append(layer['params']['weight'])
            denseB.append(layer['params']['bias'])
    return convW, convB, denseW, denseB