from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
def create_model():

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                     input_shape=(299, 299, 3)))

    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    return model


def save_conv_img(conv_img):
    feature_maps = np.squeeze(conv_img, axis=0)
    img_num = feature_maps.shape[2]  
    all_feature_maps = []
    for i in range(0, img_num):
        single_feature_map = feature_maps[:, :, i]
        all_feature_maps.append(single_feature_map)
        plt.imshow(single_feature_map)
        plt.savefig('conv_feature/' + 'feature_{}'.format(i))

    sum_feature_map = sum(feature_map for feature_map in all_feature_maps)
    plt.imshow(sum_feature_map)
    plt.savefig("conv_feature/feature_map_sum.png")


def create_dir():
    if not os.path.exists('conv_feature'):
        os.mkdir('conv_feature')
    else:
        shutil.rmtree('conv_feature')
        os.mkdir('conv_feature')


if __name__ == '__main__':
    img = cv2.imread('./wang.png')
    print(img.shape)

    create_dir()

    model = create_model()

    img_batch = np.expand_dims(img, axis=0)

    conv_img = model.predict(img_batch)  
    save_conv_img(conv_img)
