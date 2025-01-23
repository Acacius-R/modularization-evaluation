# **Modularization**

本项目旨在复现模型模块化相关论文提到的模块化方法 并对其进行重新评估

---

## **文件组织结构**

以下是项目目录及其文件功能的详细说明：

- **`decomposeDNNintoModules/`**  
  《On Decomposing a Deep Neural Network into Modules》中提出的模块化方法，参照decomposeDNNintoModules库(https://github.com/rangeetpan/decomposeDNNintoModules/tree/master)

- **`gradsplitter/`**  
  《Reusing Convolutional Neural Network Models through Modularization and Composition》中提出的模块化方法Gradsplitter,参考Gradsplitter库（https://github.com/qibinhang/GradSplitter/tree/main）




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
