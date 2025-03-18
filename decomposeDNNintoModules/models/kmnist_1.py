import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical # type: ignore
import numpy as np


dataset, info = tfds.load("kmnist", as_supervised=True, with_info=True)
x_train = np.array([example[0].numpy() for example in dataset["train"]])
y_train = np.array([example[1].numpy() for example in dataset["train"]])
x_test = np.array([example[0].numpy() for example in dataset["test"]])
y_test = np.array([example[1].numpy() for example in dataset["test"]])


num_classes = info.features["label"].num_classes
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.astype('float32') / 255  # Normalize the pixel values
x_test = x_test.astype('float32') / 255  # Normalize the pixel values


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Flatten(input_shape=(img_rows, img_cols, 1), name="Input"),
    layers.Dense(49, activation='relu', name="H1"), 
    layers.Dense(num_classes, activation='softmax', name="Output") 
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
root_dir = '/bdata/rq/modularization/decomposeDNNintoModules/'
save_dir = root_dir + 'models/KMNIST_1.h5'

test_loss, test_acc = model.evaluate(x_test, y_test)
model.save(save_dir)
print(f"Test accuracy: {test_acc}")