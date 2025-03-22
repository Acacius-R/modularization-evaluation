import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras import Sequential # noqa
from scipy.io import loadmat # noqa
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D # noqa
from keras.utils import to_categorical # noqaoqa
from sklearn.model_selection import train_test_split # noqa
import numpy as np # noqa
from scipy import io as sio # noqa
def load_svhn_data():
    train_data = loadmat('incite-issta24/Models/Datasets/svhn/test_32x32.mat')
    test_data = loadmat('incite-issta24/Models/Datasets/svhn/train_32x32.mat')
    x_train = np.transpose(train_data['X'], (3, 0, 1, 2))
    y_train = train_data['y'].flatten()
    x_test = np.transpose(test_data['X'], (3, 0, 1, 2))
    y_test = test_data['y'].flatten()
    
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_svhn_data()
    y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
    x_train, x_test = x_train.astype('float32') / 255., x_test.astype('float32') / 255.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # np.save('cifar10-train-imgs.npy', x_train)
    # np.save('cifar10-test-imgs.npy', x_test)
    # np.save('cifar10-train-label.npy', y_train)
    # np.save('cifar10-test-label.npy', y_test)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    model.save('VGGNet16-SVHN.keras')