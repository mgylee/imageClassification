from tensorflow import keras
import numpy
import os


# NN Class
class ImageClassModel:
    def __init__(self, labels):
        # Create inference array
        self.inference_array = numpy.array(labels)

        # Create model
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', activation = 'relu', input_shape = (28, 28, 3)))
        self.model.add(keras.layers.Conv2D(64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Conv2D(128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation = 'relu'))
        self.model.add(keras.layers.Dropout(0.4))
        self.model.add(keras.layers.Dense(64, activation = 'relu'))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(len(labels), activation = 'softmax'))

        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

    def fit(self, X_train, y_train = None, epochs = 1, validation_data = None):
        if y_train:
            self.model.fit(x = X_train, y = y_train, epochs = epochs, validation_data = validation_data)
        else:
            self.model.fit(x = X_train, validation_data = validation_data, epochs = epochs)

    def evaluate(self):
        # TODO: Create evaluation method
        pass

    def predict_classes(self, image_array):
        self.class_index = self.model.predict_classes(image_array)
        return self.inference_array[self.class_index[0]]

    def save_model(self, path):
        model_path = os.path.join(path, 'model.h5')
        label_path = os.path.join(path, 'label.txt')
        self.model.save(model_path)
        numpy.savetxt(label_path, self.inference_array, fmt = '%s')