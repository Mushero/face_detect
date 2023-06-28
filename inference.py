import numpy as np
import dataset
import cv2
from keras import backend as K
import vggmodel
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
K.set_image_data_format('channels_last')

if __name__ == "__main__":
    model = vggmodel.VGG16(5)
    model.load_weights("./weights/best.h5")
    img = cv2.imread("./wang.png")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    img = np.expand_dims(img,axis = 0)
    img = dataset.resize_image(img,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print(dataset.print_answer(np.argmax(model.predict(img))))