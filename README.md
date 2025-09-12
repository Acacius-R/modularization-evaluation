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

- Navigate to each modularization method directory in the code base.

- Run the four analysis scripts in each folder: `rq1.py`,`rq2.py`,`rq3.py`,`rq4.py`
- Upon completion, data will be automatically saved in the analysis/ subfolder within each respective directory.


## **Dependencies and Environment**
Before running the code, please ensure the following dependencies are installed:
- Python >= 3.12
- PyTorch >= 2.4
- Tensoofow-2.16.2 >= 2.4
- NumPy
- Matplotlib

