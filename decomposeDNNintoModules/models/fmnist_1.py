import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import fashion_mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np

# laod mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
num_classes = len(tf.unique(y_train)[0].numpy())
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
# print(img_rows, img_cols)


x_train = x_train.astype('float32') / 255  # Normalize the pixel values
x_test = x_test.astype('float32') / 255  # Normalize the pixel values

# convert into onehot
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Flatten(input_shape=(img_rows, img_cols, 1), name="Input"),
    layers.Dense(49, activation='relu', name="H1"),    
    layers.Dense(num_classes, activation='softmax', name="Output") 
])

# complie model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
root_dir = '/bdata/rq/modularization/decomposeDNNintoModules/'
save_dir = root_dir + 'models/FMNIST_1.h5'
# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
model.save(save_dir)
print(f"Test accuracy: {test_acc}")