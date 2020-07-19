import keras
from keras.preprocessing import image
import numpy as np

# Convert list of training images to array
def preprocess(img):
    list = []
    b = image.load_img(img, target_size=(28,28,3))
    b = image.img_to_array(b)
    b = b/255
    list.append(b)
    list = np.array(list)
    return(list)