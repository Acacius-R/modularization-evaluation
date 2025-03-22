from keras.models import load_model,clone_model# type: ignore
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist,fashion_mnist # type: ignore
import csv
import os
import tensorflow_datasets as tfds
from utils import *
from scipy.stats import ttest_rel
from tensorflow.keras import layers, models # type: ignore
from concern_modularzation import cm_rie
from tensorflow.keras import backend as K # type: ignore
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
def module_predict(modules):
    
    predict = np.zeros((len(y_test), 10))
    for i in range(10):
        tmp = []
        module_to_eval = modules[i]
        outputs = module_to_eval.predict(x_test)
        predict[:, i] = outputs[:, i]

    labels = np.argmax(y_test, axis=1)
    predict = np.argmax(predict, axis=1)
    correct = np.sum(labels == predict)
    accuracy = correct / len(labels)
    
    print("Accuracy: ", accuracy)
    return accuracy
        

def accuracy_evaluate(modules):

    grouped_data_x = {i: [] for i in range(10)} 
    grouped_data_y = {i: [] for i in range(10)} 
    for i in range(len(y_test)):
        label = np.argmax(y_test[i])
        grouped_data_x[label].append(x_test[i])
        grouped_data_y[label].append(y_test[i])


    result = []
    for i in range(10):
        module_to_eval = modules[i]
        tmp = []
        for j in range(10):
            x_subset = grouped_data_x[j]
            y_subset = grouped_data_y[j]
            loss, acc = module_to_eval.evaluate(np.array(x_subset), np.array(y_subset))
            tmp.append(acc)
        result.append(tmp)

    with open(f'{output_dir}/accuracycsv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

def accuracy_evaluate_v2(modules):

    grouped_data_x = {i: [] for i in range(10)}  
    grouped_data_y = {i: [] for i in range(10)} 
    for i in range(len(y_test)):
        label = y_test[i]
        grouped_data_x[label].append(x_test[i])
        grouped_data_y[label].append(label)

    for i in range(10):
        other_labels_x = []
        other_labels_y = []
        for j in range(10):
            if j != i:
                other_labels_x.extend(grouped_data_x[j])
                other_labels_y.extend(grouped_data_y[j])

        num_samples = len(grouped_data_x[i])
        if num_samples > 0:
            indices = np.random.choice(len(other_labels_x), num_samples, replace=False)
            grouped_data_x[i].extend([other_labels_x[idx] for idx in indices])
            grouped_data_y[i].extend([other_labels_y[idx] for idx in indices])


    result = []
    for i in range(10):
        module_to_eval = modules[i]
        tmp = []
        for j in range(10):
            x_subset = grouped_data_x[j]
            y_subset = grouped_data_y[j]
            outputs = module_to_eval.predict(np.array(x_subset))
            # outputs[:,i]>0.5 
            predict  = np.argmax(outputs, axis=1)
            predict = (predict==i).astype(int)
            y_subset=np.array(y_subset)
            y_subset = (y_subset==j).astype(int)
            correct = np.sum(y_subset == predict)
            acc = correct / len(y_subset)
            tmp.append(acc)
        result.append(tmp)

    with open(f'{output_dir}/accuracy_v2_random.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

def similarity_evaluate():

    def jaccard_similarity(list1, list2):
        intersection = len(np.intersect1d(list1, list2))
        union = len(np.union1d(list1, list2))
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
            # print("Module ", i, " and Module ", j, " Jaccard similarity of weights: ", similarity)
        result.append(tmp)

    


    with open(f'{output_dir}/similarity.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

def random_weight_module_predict(random_seed):
    print("Random seed: ", random_seed)
    np.random.seed(random_seed)
    random_modules = []
    for i in range(10):  
        module_to_eval = clone_model(module[i])  
        module_to_eval.set_weights(module[i].get_weights())
        weights, biases = get_weights_and_biases(module_to_eval)
        orig_weights, orig_biases = get_weights_and_biases(orig_model)

        num_nonzero_weights = sum(np.count_nonzero(w) for w in weights[:-1])
        num_nonzero_biases = sum(np.count_nonzero(b) for b in biases[:-1])
        
       
        all_weights = np.concatenate([w.flatten() for w in weights[:-1]])
        all_biases = np.concatenate([b.flatten() for b in biases[:-1]])
        
        
        random_weights = np.concatenate([w.flatten() for w in orig_weights[:-1]]).copy()
        random_biases = np.concatenate([b.flatten() for b in orig_biases[:-1]]).copy()
        
        zero_indices_weights = np.random.choice(np.arange(random_weights.size), size=random_weights.size - num_nonzero_weights, replace=False)
        zero_indices_biases = np.random.choice(np.arange(random_biases.size), size=random_biases.size - num_nonzero_biases, replace=False)
        
        random_weights[zero_indices_weights] = 0
        random_biases[zero_indices_biases] = 0
        
        
        start_idx = 0
        for j in range(len(weights) - 1):
            end_idx = start_idx + weights[j].size
            weights[j] = random_weights[start_idx:end_idx].reshape(weights[j].shape)
            start_idx = end_idx
        
        start_idx = 0
        for j in range(len(biases) - 1):
            end_idx = start_idx + biases[j].size
            biases[j] = random_biases[start_idx:end_idx].reshape(biases[j].shape)
            start_idx = end_idx
        
       
        for j in range(len(weights) - 1):
            module_to_eval.layers[j+1].set_weights([weights[j], biases[j]])
        D,b = get_weights_and_biases(module_to_eval)
        D,b = cm_rie(D,b,i)

        for j in range(1,len(module_to_eval.layers)):
            module_to_eval.layers[j].set_weights([D[j-1],b[j-1]])
        
        module_to_eval.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        random_modules.append(module_to_eval)

    accuracy_evaluate_v2(random_modules)
    # accuracy_evaluate(random_modules)
    # module_predict(random_modules)
    
    # return acc


def paired_t_test(n):
    random_acc = []
    global random_seed
    for i in range(n):
        random_acc.append(random_weight_module_predict(random_seed))
        random_seed+=1

    orig_acc = module_predict(module)

    t_statistic, p_value = ttest_rel([orig_acc]*n, random_acc)

 

def evaluate_per_class(model, x_test, y_test, num_classes, output_dir):

    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}


    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)


    for i in range(len(y_true_classes)):
        label = y_true_classes[i]
        pred = y_pred_classes[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    results = []
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            results.append(f"Accuracy of class {i}: {accuracy:.2f}%")
        else:
            results.append(f"Accuracy of class {i}: N/A (no samples)")

    with open(f'{output_dir}/model_acc.txt', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')


def calculate_params_and_flops(model, input_shape):

    input_tensor = tf.keras.Input(shape=input_shape[1:])
    model_output = model(input_tensor)
    model_for_flops = tf.keras.Model(inputs=input_tensor, outputs=model_output)
    
    total_params = model.count_params()
    

    concrete_func = tf.function(lambda x: model_for_flops(x)).get_concrete_function(tf.TensorSpec([1] + list(input_shape[1:]), model_for_flops.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        total_flops = flops.total_float_ops
    
    print(f"Total FLOPs: {total_flops}")
    print(f"Total Params: {total_params}")
    return total_flops, total_params
    

if __name__ == '__main__':
    labels =[0,1,2,3,4,5,6,7,8,9]
    num_classes = 10
    approach = 'Approach-CMRIE'
    model_name = 'MNIST_1'
    root_dir = '/bdata/rq/modularization/decomposeDNNintoModules/'
    output_dir = f'{root_dir}{approach}/analaysis/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    module = {i: load_model(f'{root_dir}{approach}/modularized_models/{model_name}/{i}.h5') for i in range(num_classes)}

    if "FMNIST" in model_name:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_test = x_test.astype('float32') / 255
        # y_test = to_categorical(y_test, 10)
    elif "KMNIST" in model_name:
        dataset, info = tfds.load("kmnist", as_supervised=True, with_info=True)
        x_test = np.array([example[0].numpy() for example in dataset["test"]])
        y_test = np.array([example[1].numpy() for example in dataset["test"]])
        x_test = x_test.astype('float32') / 255  # Normalize the pixel values
        # y_test = to_categorical(y_test, num_classes)
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32') / 255
        # y_test = to_categorical(y_test, 10)

    orig_model = load_model(f'{root_dir}models/{model_name}.h5')
    random_seed =302
    # acc = module_predict(module)
    # print("Accuracy: ", acc)
    similarity_evaluate()
    # accuracy_evaluate_v2(module)
    # paired_t_test(10)
    # random_weight_module_predict(302)
    # evaluate_per_class(orig_model, x_test, y_test, num_classes, output_dir)
    # flops,param_num = calculate_params_and_flops(orig_model, x_test.shape)
    # composed_flops = 0
    # composed_params = 0
    # for i in range(10):
    #     eval_module = module[i]
    #     flops,param_num = calculate_params_and_flops(eval_module, x_test.shape)
    #     composed_flops += flops
    #     composed_params += param_num

    # print(f"Composed Model FLOPs: {composed_flops}")
    # print(f"Composed Model Params: {composed_params}")
    # print("FLOPs: ", flops)
    # print("Params: ", param_num)

    
    
    


    





