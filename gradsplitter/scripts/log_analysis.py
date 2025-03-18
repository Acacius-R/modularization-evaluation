import re
import numpy as np
import matplotlib.pyplot as plt
def read_accuracy_from_log(file_path):
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'^\d\.\d{4}$', line.strip())
            if match:
                accuracies.append(float(match.group()))
    return accuracies

estimator_indices = [1, 3, 4, 9, 10, 11, 12, 14, 15, 16] 
model = 'rescnn'
dataset = 'cifar10'

random_acc = []
acc =[]
for estimator_idx in estimator_indices:
    random_file_path = f'./eval_{model}_{dataset}_estimator_{estimator_idx}_random.log'
    file_path = f'./eval_{model}_{dataset}_estimator_{estimator_idx}.log'   
    random_acc.append(read_accuracy_from_log(random_file_path))
    acc.append(read_accuracy_from_log(file_path))


random_acc = np.array(random_acc)
acc = np.array(acc)
random_acc_mean = np.mean(random_acc, axis=1)
acc_mean = np.mean(acc, axis=1)

x = np.arange(10)
width = 0.35  # 柱状图的宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, random_acc_mean, width, label='Random')
rects2 = ax.bar(x + width/2, acc_mean, width, label='gradsplitter')
ax.set_xlabel('class')
ax.set_ylabel('Mean Accuracy')
ax.legend()
ax.set_title('Random vs gradsplitter(kernal selection)')
plt.savefig('accuracy.png')