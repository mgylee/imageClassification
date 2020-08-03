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
        self.indexes = numpy.arange(len(self.image_label))

        # Shuffle dataset
        self.on_epoch_end()

    def __len__(self):
        return int(numpy.floor(len(self.image_id) / self.batch_size))

    def __getitem__(self, index):
        batch_index = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_index)
        return X, y

    def on_epoch_end(self):
        numpy.random.shuffle(self.indexes)

    # Generate next mini-batch of data
    def __data_generation(self, batch_index):
        X = numpy.empty((len(batch_index), *self.dim))
        y = numpy.empty(len(batch_index))
        for enum, index in enumerate(batch_index):
            image_path = os.path.join(self.path, 'images', self.image_id[index])
            image_arr = keras.preprocessing.image.load_img(image_path, color_mode = 'rgb', target_size = self.dim)
            image_arr = keras.preprocessing.image.img_to_array(image_arr)
            X[enum,] = image_arr
            y[enum,] = self.image_label[index]

        X = X / 255 # Normalize values
        return X, keras.utils.to_categorical(y, num_classes = self.n_classes)