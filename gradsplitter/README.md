# **Gradsplitter**

本项目旨在复现论文《Reusing Convolutional Neural Network Models through Modularization and Composition》中提出的模块化方法Gradsplitter，并对深度神经网络（DNN）进行模块化处理、重用和性能评估。本项目参考Gradsplitter库（https://github.com/qibinhang/GradSplitter/tree/main）

---

## **文件组织结构**

以下是项目目录及其文件功能的详细说明：

- **`data/`**  
  用于保存模型训练和测试数据和训练好的模型。

- **`models/`**  
  模型架构代码。

- **`cofigures/`**  
  模型相关参数配置文件。

- **`scripts/`**  
  运行实验的脚本，运行实验的日志。

- **`experiments/`**  
  评估模块化模型。（添加重训练全连接层的步骤）。

- **`splitter/`**  
  特定模型的模块化工具。

- **`utils/`**  
  模块化和重用过程中会使用到的函数。（对module_tools.py文件中的get_target_module_info_for_rescnn()函数进行修改以支持随机选择的对比）。

- **`grad_splitter.py`**  
  进行模块化

- **`train.py`**  
  训练模型



---

## **依赖库和环境**
在运行代码前，请确保已安装以下依赖：
- Python >= 3.12
- PyTorch >= 2.4
- NumPy
- Matplotlib

安装所有依赖：
```bash
pip install -r requirements.txt
```
