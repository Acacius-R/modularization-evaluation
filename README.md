# **Modularization**

we aim to reproduct the modularization method and evaluate them

---

## **文件组织结构**

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



---

## **依赖库和环境**
在运行代码前，请确保已安装以下依赖：
- Python >= 3.12
- PyTorch >= 2.4
- -tensoofow-2.16.2 >= 2.4
- NumPy
- Matplotlib

安装所有依赖：
```bash
pip install -r requirements.txt
```
