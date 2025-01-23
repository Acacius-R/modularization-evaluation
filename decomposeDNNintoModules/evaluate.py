from keras.models import load_model
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import csv
import os
from utils import *
labels =[0,1,2,3,4,5,6,7,8,9]
num_classes = 10
approach = 'Approach-TI'
model_name = 'MNIST_1'
output_dir = f'{approach}/analaysis/{model_name}'
os.makedirs(output_dir, exist_ok=True)
module = {i: load_model(f'{approach}/modularized_models/{model_name}/{i}.h5') for i in range(num_classes)}

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

    with open(f'{output_dir}/accuracy.csv', 'w', newline='', encoding='utf-8')as f:
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

    orig_model = load_model(f'./models/{model_name}.h5')
    weights,biases = get_weights_and_biases(orig_model)
    weights = [w.flatten() for w in weights]
    weights= [np.array(w) for w in weights]
    weights= [np.array(w) for w in weights]
    weights = np.concatenate(weights)
    y = np.count_nonzero(weights)

    for i in range(10):
        similarity = jaccard_similarity(weights, weights_to_compare[i])
        print("Original model and Module ", i, " Jaccard similarity of weights: ", similarity)
        x = np.count_nonzero(weights_to_compare[i])
        print("Module ", i, " weights percentage: ", x/y)

        
    with open(f'./{output_dir}/similarity.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

# def weights_percentage_evaluate():

#     orig_model = load_model(f'./models/{model_name}.h5')
#     weights,biases = get_weights_and_biases(orig_model)
    
#     total = np.count_nonzero(weights)
#     for i in range(10):
#         module_to_eval = module[i]
#         w,b = get_weights_and_biases(module_to_eval)
#         x = np.count_nonzero(w)
#         print("Module ", i, " weights percentage: ", x/total)


if __name__ == '__main__':
    similarity_evaluate()
    # accuracy_evaluate()


    





