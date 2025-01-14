import numpy as np
from keras.models import load_model
from tensorflow.keras.datasets import mnist
from tangling_identification import tangling_identification
from concern_identification import concern_identification
from utils import *
from concern_modularzation import cm_c

labels =[0,1,2,3,4,5,6,7,8,9]
data_dir = '/home/rq/modularization/decomposeDNNintoModules/datasets/MNIST'
slicing_model = load_model('/home/rq/modularization/decomposeDNNintoModules/models/MNIST_1.h5')
model = load_model('/home/rq/modularization/decomposeDNNintoModules/models/MNIST_1.h5')



#加载训练数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data ={i:[] for i in labels}
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
for i in range(len(y_train)):
    train_data[y_train[i]].append(x_train[i])
# for i in labels:
#     data_name = data_dir + '/train_data_' + str(i) + '.npy'
#     train_data[i] = np.load(data_name)

for i in labels:
    print("#Module "+str(i)+" in progress....")
    
    weights,biases = get_weights_and_biases(slicing_model)
    D = [np.zeros_like(w) for w in weights]
    b = [np.zeros_like(b) for b in biases]
    
    inputs = train_data[i][0:1000]
    indicator = True
    for input in inputs:
        D,b = concern_identification(slicing_model, input, indicator, D, b)
        if np.count_nonzero(D[-1]) <315:
            indicator = True
        else:
            indicator = False
    
    untarget_data = []
    for j in labels:
        if j != i:
            untarget_data.extend(train_data[j][:2])
    np.random.shuffle(untarget_data)
    for input in untarget_data:
        D,b = tangling_identification(slicing_model, input, indicator, D, b)
        if np.count_nonzero(D[-1]) <315:
            indicator = True
        else:
            indicator = False
    
    D,b = cm_c(D,b,i)
    

    model.layers[1].set_weights([D[0],b[0]])
    model.layers[2].set_weights([D[1],b[1]])
    model.save('./modularized_models/CMC/'+str(i)+'.h5')