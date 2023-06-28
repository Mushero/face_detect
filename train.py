from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
import tensorflow as tf
import vggmodel
import numpy as np
import dataset
import cv2
from keras import backend as K

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
K.set_image_data_format('channels_last')


def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread("./dataset/images/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            i = (i + 1) % n
        X_train = dataset.resize_image(X_train, (224, 224))
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=5)
        yield (X_train, Y_train)


if __name__ == "__main__":

    log_dir = "./weights/"

    with open("./dataset/train.txt", "r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * 0.3)
    num_train = len(lines) - num_val

    model = vggmodel.VGG16(5)

    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    batch_size = 8

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=100,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir + 'best.h5')

