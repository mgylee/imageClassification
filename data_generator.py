from tensorflow import keras
import numpy
import os


# Data generator to stream data for model training
class ImageGenerator(keras.utils.Sequence):
    def __init__(self, image_id, image_label, batch_size = 32, dim = (28, 28, 3), n_classes = 10, path = './data'):
        self.image_label = image_label
        self.image_id = image_id
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.path = path

    def __len__(self):
        return int(numpy.floor(len(self.image_id) / self.batch_size))

    def __getitem__(self, index):

        batch_image = self.image_id[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label = self.image_label[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation(batch_image, batch_label)
        return X, y

    # Generate next mini-batch of data
    def __data_generation(self, batch_image, batch_label):
        X = numpy.empty((self.batch_size, *self.dim))
        for i, name in enumerate(batch_image):
            image_path = os.path.join(self.path, 'images', name)
            image_arr = keras.preprocessing.image.load_img(image_path, color_mode = 'rgb', target_size = self.dim)
            image_arr = keras.preprocessing.image.img_to_array(image_arr)
            X[i,] = image_arr

        return X, keras.utils.to_categorical(batch_label, num_classes = self.n_classes)