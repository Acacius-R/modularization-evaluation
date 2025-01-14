import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 文件
approach = "Approach-CMRIE"
model_name = "MNIST_1"
file_path = f"{approach}/analaysis/{model_name}/"
file_name = "similarity.csv"
data = pd.read_csv(file_path+file_name, header=None)  # 假设文件没有列名

# 确保数据为数值类型
data = data.astype(float)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".1f", cmap="Blues", cbar=True, square=True)

# 添加标题
plt.title("similarity")
plt.savefig(file_path+"similarity.png")

file_name = "accuracy.csv"
data = pd.read_csv(file_path+file_name, header=None)  # 假设文件没有列名

# 确保数据为数值类型
data = data.astype(float)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".1f", cmap="Blues", cbar=True, square=True)

# 添加标题
plt.title("accuracy")
plt.savefig(file_path+"accuracy.png")