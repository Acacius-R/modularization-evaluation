import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = len(tf.unique(y_train)[0].numpy())
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
# print(img_rows, img_cols)

# 数据预处理
# x_train = x_train.reshape((x_train.shape[0], 28 * 28))  # Flatten the images
# x_test = x_test.reshape((x_test.shape[0], 28 * 28))  # Flatten the images
x_train = x_train.astype('float32') / 255  # Normalize the pixel values
x_test = x_test.astype('float32') / 255  # Normalize the pixel values

# 将标签转换为one-hot编码
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

train_data_by_label = {i: [] for i in range(10)}  # 创建一个字典来存储按标签分组的数据
for img, label in zip(x_train, y_train):
    label = np.argmax(label)
    train_data_by_label[label].append(img)

# 将数据按标签分组：测试集
test_data_by_label = {i: [] for i in range(10)}  # 创建一个字典来存储按标签分组的数据
for img, label in zip(x_test, y_test):
    label = np.argmax(label)
    test_data_by_label[label].append(img)

for label in range(10):
    train_data_by_label[label] = np.array(train_data_by_label[label])
    np.save(f'./datasets/MNIST/train_data_{label}.npy', train_data_by_label[label])
    test_data_by_label[label] = np.array(test_data_by_label[label])
    np.save(f'./datasets/MNIST/test_data_{label}.npy', test_data_by_label[label])

# 构建DNN模型
model = models.Sequential([
            layers.Flatten(input_shape=(img_rows, img_cols,1), name = "Input"),
            layers.Dense(49, activation='relu' ,name = "H"),
            layers.Dense(num_classes, activation='softmax', name = "output")
        ])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
model.save('./models/MNIST_1.h5')
print(f"Test accuracy: {test_acc}")