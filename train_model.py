from sklearn import preprocessing
from sklearn import model_selection
from tensorflow import keras
from package import nn_model
from package import data_generator
import argparse
import csv
import pandas
import numpy
import os


def import_data(path = './data'):
    dataset = pandas.read_csv(os.path.join(path, 'data_labels.csv'), names = ['image_id', 'label'])
    dataset = dataset.to_numpy()
    numpy.random.shuffle(dataset)
    split_data = numpy.hsplit(dataset, 2)

    image_id = numpy.ravel(split_data[0])
    image_label, image_classes = encode_labels(numpy.ravel(split_data[1]))

    return image_id, image_label, image_classes

def encode_labels(label):
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(label)
    
    return label_encoder.transform(label), label_encoder.classes_


if __name__ == '__main__':
    image_id, image_label, image_classes = import_data()
    gparams = {
        'dim': (28, 28, 3),
        'batch_size': 64,
        'n_classes': len(image_classes),
    }

    X_train, X_test, y_train, y_test = model_selection.train_test_split(image_id, image_label, test_size = 0.2)

    image_train = data_generator.ImageGenerator(X_train, y_train, **gparams)
    image_val = data_generator.ImageGenerator(X_test, y_test, **gparams)

    image_model = nn_model.ImageClassModel(image_classes)
    image_model.fit(X_train = image_train, validation_data = image_val, epochs = 5)
    image_model.save_model(path = './model')