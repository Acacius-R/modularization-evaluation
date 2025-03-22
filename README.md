# **Modularization**

we aim to reproduct the modularization method and evaluate them

---

## **File organization**

folloing is the statement of files

- **`decomposeDNNintoModules/`**  
  corresponding to the method in ` On Decomposing a Deep Neural Network into Modules`denoted as cmrie in our paper(https://github.com/rangeetpan/decomposeDNNintoModules/tree/master)

- **`gradsplitter/`**  
  corresponding to the method in `Reusing Convolutional Neural Network Models through Modularization and Composition`(https://github.com/qibinhang/GradSplitter/tree/main)
- **`cnnsplitter/`**  
  corresponding to the method in`Patching Weak Convolutional Neural Network Models through Modularization and Composition`(https://github.com/qibinhang/CNNSplitter/tree/main)
- **`mwt/`**  
  corresponding to the method in`Modularizing while Training: A New Paradigm for Modularizing DNN Models`(https://github.com/qibinhang/MwT)
- **`decomposeWithMask/`**  
  corresponding to the method in`Neural Network Module Decomposition and Recomposition with Superimposed Masks`

To conduct our experiment:
- **`decomposeDNNintoModules/`**  
  run the accuracy_evaluate() and similarity_eavluate() in `evaluate.py `to get the result of RQ1 and RQ2, calculate_params_and_flops() to get result of RQ3 and random_weight_module_predict() to get result of RQ4
- **`gradsplitter/`**  
  run` evaluate_modules.py` according to comments to get the result of RQ1 RQ2 RQ3 RQ4

- **`cnnsplitter/`**  
  run the  `evaluate_binary.py `to get the result of RQ1 and RQ2, calculate_params_and_flops() in `evaluate.py` to get result of RQ3 and compare_with_random() to get result of RQ4

- **`mwt/`**  
  run the calculate_similarity() and calculate_cross_accuracy() `mdoularizer.py `to get the result of RQ1 and RQ2, calculate_params() to get result of RQ3 and set `random_selection = True` in  generate_target_module()to get random modules 

- **`decomposeWithMask/`**  
  run the evaluate_module_per_class() and calculate_jaccard_similarity() in `evalute.py `to get the result of RQ1 and RQ2, run calculate_flops to get result of RQ3 and generate_random_modules()to get random modules 

- **`incite/`**  
  run the accuracy_evaluate() and similarity_evaluate() in `combiner.py `to get the result of RQ1 and RQ2, run calculate_params_and_flops() to get result of RQ3 and random_weight_module_generate()to get random modules 
---

## **Dependencies and Environment**
Before running the code, please ensure the following dependencies are installed:
- Python >= 3.12
- PyTorch >= 2.4
- Tensoofow-2.16.2 >= 2.4
- NumPy
- Matplotlib

