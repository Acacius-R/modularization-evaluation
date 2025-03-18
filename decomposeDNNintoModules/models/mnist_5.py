import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

ds_train, ds_test = tfds.load('emnist/balanced', split=['train', 'test'], as_supervised=True, batch_size=-1)
x_train, y_train = tfds.as_numpy(ds_train)
x_test, y_test = tfds.as_numpy(ds_test)

num_classes = len(np.unique(y_train))
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.astype('float32') / 255  # Normalize the pixel values
x_test = x_test.astype('float32') / 255  # Normalize the pixel values

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Flatten(input_shape=(img_rows, img_cols, 1), name="Input"),
    layers.Dense(49, activation='relu', name="H1"), 
    layers.Dense(49, activation='relu', name="H2"),   
    layers.Dense(49, activation='relu', name="H3"),   
    layers.Dense(num_classes, activation='softmax', name="Output") 
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
root_dir = '/bdata/rq/modularization/decomposeDNNintoModules/'
save_dir = root_dir + 'models/MNIST_5.h5'

test_loss, test_acc = model.evaluate(x_test, y_test)
model.save(save_dir)
print(f"Test accuracy: {test_acc}")