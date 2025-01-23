# **Modularization**

本项目旨在复现论文《On Decomposing a Deep Neural Network into Modules》中提出的模块化方法，并对深度神经网络（DNN）进行模块化处理、重用和性能评估。

---

## **文件组织结构**

以下是项目目录及其文件功能的详细说明：

- **`datasets/`**  
  用于保存模型训练和测试数据（目前未实际使用）。

- **`models/`**  
  保存训练好的模型（待模块化的模型）。命名规则为：`训练集名称` + `_` + `隐藏层数量`。

- **`modularized_models/`**  
  保存模块化后的模型。

- **`Approach-XXX/`**  
  保存使用xxx方法模块化后的模型和分析结果

- **`modularize.py`**  
  按照 `decomposeDNNintoModules` 中的综合使用CI、TI、CM方法进行模型模块化。

- **`reuse.py`**  
  对应 `decomposeDNNintoModules` 中的方法 `1TIreuse.py` 的实现（对源代码进行了优化以提高运行效率）。

- **`evaluate.py`**  
  包含模块化后模型的准确率测试与相似性测试的功能。

- **`concern_identification.py`**  
  实现 CI（Concern Identification）。

- **`concern_modularization.py`**  
  实现 CM（Concern Modularization）。

- **`tangling_identification.py`**  
  实现 TI（Tangling Identification）。

---

## **依赖库和环境**
在运行代码前，请确保已安装以下依赖：
- Python >= 3.7
- PyTorch >= 1.8
- NumPy
- Matplotlib

安装所有依赖：
```bash
pip install -r requirements.txt
```

decomposeDNNintoModules中参照decomposeDNNintoModules库(https://github.com/rangeetpan/decomposeDNNintoModules/tree/master) 复现论文《On decomposing a deep neural network into modules》模块化方法

对模型进行模块运行modularize_ti.py文件即可，reuse.py是重用文件，evaluate.py对模块化后模型进行评估