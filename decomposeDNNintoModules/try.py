from keras.models import load_model
model = load_model('./models/MNIST_1.h5')
weights = [layer.get_weights()[0] for layer in model.layers[1:]]
last_layer_weights = weights[-1]
print(last_layer_weights[1])
x1 = last_layer_weights[1]
x1[5]=0
print(last_layer_weights[1])