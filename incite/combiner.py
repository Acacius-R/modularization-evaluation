from ast import mod
from glob import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from argparse import ArgumentParser  # noqa
from keras.models import load_model,clone_model # noqa
import numpy as np  # noqa
from os import path # noqa
import pandas as pd
from keras.datasets import cifar10
from keras.utils import to_categorical #
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import csv
from util import *
def random_weight_module_generate(random_seed,origin_model,modules):
    print("Random seed: ", random_seed)
    np.random.seed(random_seed)
    random_modules = []
    for i in range(len(modules)):  
        train_inputs ,train_outputs = binarize_and_balance(train_input_data,train_output_data,i)
        module_to_eval = clone_model(modules[i])  
        module_to_eval.set_weights(modules[i].get_weights())

        module_to_eval = clone_model(modules[i])

        module_to_eval.set_weights(modules[i].get_weights())
        

        for i in range(len(module_to_eval.layers)-1):
            layer = module_to_eval.layers[i]
            origin_layer = origin_model.layers[i]
            new_weights = []
            
            for weight_idx, w in enumerate(layer.get_weights()):

                origin_w = origin_layer.get_weights()[weight_idx]
                
                flat_origin = origin_w.flatten()
                selected = np.random.choice(
                    flat_origin, 
                    size=w.size, 
                    replace=False
                )
                
                new_weights.append(selected.reshape(w.shape))
            layer.set_weights(new_weights)

        module_to_eval.fit(train_inputs,
                            train_outputs,
                            epochs=128,
                            validation_split=0.1,
                            )
        
        module_to_eval.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        module_to_eval.save(f'module_random{i}.keras')
        random_modules.append(module_to_eval)
    return random_modules


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
    

    return total_flops, total_params

def accuracy_evaluate(modules):

    grouped_data_x = {i: [] for i in range(10)}  
    grouped_data_y = {i: [] for i in range(10)} 
    for i in range(len(true_labels)):
        label = true_labels[i]
        grouped_data_x[label].append(input_data[i])
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

    with open(f'accuracy.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

def similarity_evaluate(module_weights):

    def jaccard_similarity(list1, list2):
        intersection = len(np.intersect1d(list1, list2))
        union = len(np.union1d(list1, list2))
        return float(intersection) / union
    
    weights_to_compare = []

    for i in range(len(module_weights)):
        weights = module_weights[i]
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
    with open(f'similarity.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-b',
                        '--basedir',
                        dest='base_dir',
                        help='Base directory of the model',
                        default='/home/rq/incite-issta24/Models/',
                        required=True)
    parser.add_argument('-i',
                        '--test-in',
                        dest='test_in_filename',
                        help='Test input file in .npy format',
                        required=True)
    parser.add_argument('-o',
                        '--test-out',
                        dest='test_out_filename',
                        help='Test output file in .npy format',
                        required=True)
    parser.add_argument('-ti',
                        '--train-in',
                        dest='train_in_filename',
                        help='Train input file in .npy format',
                        required=True)
    parser.add_argument('-to',
                        '--train-out',
                        dest='train_out_filename',
                        help='Train output file in .npy format',
                        required=True)
    parser.add_argument('-th',
                        '--threshold',
                        dest='threshold',
                        default='0.6',
                        help='Percentage of top most impactful neuron clusters to be selected from each layer '
                             '(default: 0.6)',
                        required=False)
    parser.add_argument('-w',
                        '--weight-fraction',
                        dest='weight_fraction',
                        default='0.1',
                        help='The fraction to add/subtract when mutating the weights (default: 0.1)',
                        required=False)
    parser.add_argument('-c',
                        '--clusters-size',
                        dest='clusters_sz',
                        default='10',
                        help='Number of neurons per cluster (default: 2).',
                        required=False)
    num_classes = 10
    args = parser.parse_args()
    basename = path.basename(args.base_dir)
    module_files = glob(path.join(args.base_dir,
                                  '*.keras'))
    modules = dict()
    module_weights = dict()
    # print(module_files)
    for module_file in module_files:
        module_index = int(path.basename(module_file).split('-')[1].split('.')[0])
        print('Loading module model for class %d...' % module_index)
        modules[module_index] = load_model(module_file)
        module_weights[module_index] = modules[module_index].get_weights()
    
    similarity_evaluate(module_weights)
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
    # x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.
    print('Loading original model...')
    original_model = load_model('/incite-issta24/Models/VGGNet16-CIFAR10/VGGNet16-CIFAR10.keras')
    print('Loading datasets 1/2...')
    input_data = np.load(args.test_in_filename)
    # input_data = x_test
    print('Loading datasets 2/2...')
    output_data = np.load(args.test_out_filename)
    train_input_data = np.load(args.train_in_filename)
    train_output_data = np.load(args.train_out_filename)
    # output_data = y_test
    accuracy_evaluate(modules)
    input_shape = input_data[0].shape
    modules_flops = module_param = 0
    for module in modules:
        flops,param = calculate_params_and_flops(module,input_shape)
        modules_flops += flops
        modules_param+=param
    print(f"Modules FLOPs:{modules_flops},Modules param:{modules_param}")
    flops,param=calculate_params_and_flops(original_model,input_shape)
    print(f"Model FLOPs:{flops},Model param:{param}")

    print('Calculating train accuracy of the original model...')
    original_acc = original_model.evaluate(input_data, output_data)[1]
    predictions = np.zeros(output_data.shape)
    results = pd.DataFrame(columns=['module_index','model_result0', 'model_result1','original_model_result0','original_model_result1'])
    for module_index, module_model in modules.items():
        print('Applying module model %d on the test dataset...' % module_index)
        x = module_model.predict(input_data, verbose=0)[:, 0]
        predictions[:, module_index] = x
    
    true_labels = np.argmax(output_data, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == true_labels)
    combined_accuracy = correct_predictions / len(true_labels)
    print('---------------')
    print('Combined Acc: %f' % combined_accuracy)
    print('Original Model Acc: %f' % original_acc)

