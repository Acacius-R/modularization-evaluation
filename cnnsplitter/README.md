# Patching Weak Convolutional Neural Network Models through Modularization and Composition



## How to modularize a trained CNN model
1. modify `global_configure.py` to set the `root_dir`.
2. run `python train.py --model simcnn --dataset cifar10` to get the pre-trained model `SimCNN-CIFAR`.
3. run `python kernel_importance_analyzer.py --model simcnn --dataset cifar10` in directory `preprocess/` to get the importance of each kernel in `SimCNN-CIFAR`.
4. run `python run_layer_sensitivity_analyzer.py --model simcnn --dataset cifar10` in directory `scripts/` to analyze the sensitivity of `SimCNN-CIFAR`.
5. modify `configures/simcnn_cifar10.py` to set the configures of GA searching.
6. run `python module_recorder.py --model simcnn --dataset cifar10`.
7. run `python run_explorer.py` 

## How to evaluate a modularized CNN model
1. modify `simcnn_cifar10.py` according to the modularization results.
2. run `evaluate.py` 


