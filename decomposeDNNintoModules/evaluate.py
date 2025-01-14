from keras.models import load_model
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import csv
from utils import *
labels =[0,1,2,3,4,5,6,7,8,9]

module = {}
for i in labels:
    module[i] = load_model('/home/rq/modularization/decomposeDNNintoModules/modularized_models/TI/'+str(i)+'.h5')

def accuracy_evaluate():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32') / 255  # Normalize the pixel values
    y_test = to_categorical(y_test, 10)

    grouped_data_x = {i: [] for i in range(10)}  # 创建空字典分组
    grouped_data_y = {i: [] for i in range(10)} 
    for i in range(len(y_test)):
        label = np.argmax(y_test[i])
        grouped_data_x[label].append(x_test[i])
        grouped_data_y[label].append(y_test[i])


    result = []
    for i in range(10):
        module_to_eval = module[i]
        tmp = []
        for j in range(10):
            x_subset = grouped_data_x[j]
            y_subset = grouped_data_y[j]
            loss, acc = module_to_eval.evaluate(np.array(x_subset), np.array(y_subset))
            tmp.append(acc)
        result.append(tmp)

    with open('accuracy.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

def similarity_evaluate():

    def jaccard_similarity(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union
    
    weights_to_compare = []

    for i in labels:
        module_to_eval = module[i]
        weights,biases = get_weights_and_biases(module_to_eval)
        weights = [w.flatten() for w in weights]

        weights= [np.array(w) for w in weights]

        weights_to_compare.append(np.concatenate(weights))
    
    result = []
    for i in range(10):
        tmp = []
        for j in range(10):
            similarity = jaccard_similarity(weights_to_compare[i], weights_to_compare[j])
            tmp.append(similarity)
            print("Module ", i, " and Module ", j, " Jaccard similarity of weights: ", similarity)
        result.append(tmp)

    orig_model = load_model('/home/rq/modularization/decomposeDNNintoModules/models/MNIST_1.h5')
    weights,biases = get_weights_and_biases(orig_model)
    weights = [w.flatten() for w in weights]
    weights= [np.array(w) for w in weights]
    weights= [np.array(w) for w in weights]
    weights = np.concatenate(weights)

    for i in range(10):
        similarity = jaccard_similarity(weights, weights_to_compare[i])
        print("Original model and Module ", i, " Jaccard similarity of weights: ", similarity)

        
    with open('simlarity.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

if __name__ == '__main__':
    similarity_evaluate()
    accuracy_evaluate()

    





