from sklearn import preprocessing
from sklearn import model_selection
from keras import preprocessing
from tqdm import tqdm
import argparse
import csv
import pandas
import numpy
import os


def norm_image(image):
    image = preprocessing.image.img_to_array(image) / 255
    return image

def import_data(path = './prep_data'):
    dataset = pandas.read_csv(os.path.join(path, 'data_labels.csv'), names = ['image_id', 'label'])
    dataset = dataset.to_numpy()
    numpy.random.shuffle(dataset)

    X = list()
    test_count = 0

    for i in dataset:
        test_count += 1
        image_path = os.path.join(path, 'images', i[0].strip('0'))
        try:
            image = preprocessing.image.load_img(image_path, target_size=(28,28,3))
            X.append(norm_image(image))
        except:
            image_path = os.path.join(path, 'images', '0' + i[0].strip('0'))
            X.append(norm_image(image))

        if test_count > 50:
            break
    
    

def encode_data():
    pass





if __name__ == '__main__':
    import_data()