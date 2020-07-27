import tensorflow as tf
from tensorflow import keras


class ImageModel:
    def __init__(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(32, (3, 3), (1, 1), 'same', activation = 'relu', input_shape = (28, 3, 3)))
        self.model.add(keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.4))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(6, activation='softmax'))

    def compile(self):
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

    def fit(self, X_train, y_train, epochs, validation_data):
        self.model.fit(X_train, y_train, epochs, validation_data = validation_data)

    def predict_classes(self, image_array):
        self.model.predict_classes(image_array)