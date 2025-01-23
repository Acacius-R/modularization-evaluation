from keras.models import load_model
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import json
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10
# x_test = x_test.reshape((x_test.shape[0], 28 * 28))  # Flatten the images
x_test = x_test.astype('float32') / 255  # Normalize the pixel values

# 将标签转换为one-hot编码
y_test = to_categorical(y_test, num_classes)
model_name = 'MNIST_1'
models = {i: load_model(f'./modularized_models/{model_name}/{i}.h5') for i in range(num_classes)}
finalPred= []
batch_preds = {i: models[i].predict(x_test) for i in range(num_classes)}

for idx in range(len(x_test)):
    pred = []
    for i in range(num_classes):
        pred_class = batch_preds[i][idx].argmax()
        if pred_class == i:
            pred.append(i)
        else:
            pred.append(10)
        
    if pred.count(10) == 10:  # 所有模型都未正确分类
        max_probs = [batch_preds[i][idx][i] for i in range(num_classes)]
        finalPred.append(np.argmax(max_probs))
    elif pred.count(10) < 9:  # 多个模型预测正确
        max_probs = [batch_preds[i][idx][i] for i in range(num_classes)]
        val_pred = [max_probs[i] for i in range(num_classes) if pred[i] != 10]
        finalPred.append(np.argmax(val_pred))
    else:  # 仅一个模型预测正确
        finalPred.append(pred.index(next(filter(lambda x: x != 10, pred))))


#In[]
from sklearn.metrics import accuracy_score
label = np.argmax(y_test, axis=1)
score = accuracy_score(finalPred,label)

print("Modularized Accuracy: "+str(score))
model=load_model('../models/MNIST_1.h5')
pred=model.predict(x_test)
pred=np.argmax(pred,axis=1)
origin_score=accuracy_score(pred,label)
print("Model Accuracy: "+str(origin_score))

#将准确率对比保存为json文件
result = {'Modularized Accuracy': score, 'Model Accuracy': origin_score}
with open(f'./analaysis/{model_name}/accuracy.json', 'w') as f:
    json.dump(result, f)



