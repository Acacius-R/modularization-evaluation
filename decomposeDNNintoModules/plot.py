import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = "./result/TI/"
file_name = "simlarity.csv"
data = pd.read_csv(file_path+file_name, header=None)  # 假设文件没有列名

# 确保数据为数值类型
data = data.astype(float)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, fmt=".1f", cmap="Blues", cbar=True, square=True)

# 添加标题
plt.title("simlarity")
plt.savefig(file_path+"simlarity.png")