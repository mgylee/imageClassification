import numpy
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_id, labels, batch_size = 32, dim = (28, 28, 3), n_channels = 1, n_classes = 10, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_id = list_id
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(numpy.floor(len(self.list_id) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_id_temp = [self.list_id[k] for k in indexes]
        X, y = self.__data_generation(list_id_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = numpy.arange(len(self.list_id))
        if self.shuffle == True:
            numpy.random.shuffle(self.indexes)

    def __data_generation(self, list_id_temp):
        X = numpy.empty((self.batch_size, *self.dim, self.n_channels))
        y = numpy.empty((self.batch_size), dtype = int)

        for i, id in enumerate(list_id_temp):
            X[i,] = numpy.load('data/' + id + '.npy')
            y[i] = self.labels[id]

        return X, keras.utils.to_categorical(y, num_classes = self.n_classes)