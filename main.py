import argparse
import csv
import os
import shutil


def check_args(args):
    assert args.unroll_data and not (args.test_path or args.train_path), "Provide --unroll-data only or --train-path and --test-path."

def check_file(filepath, label_class):
    label = filepath.split('/')[-1]

    if label in label_class:
        return label
    else:
        return False

# Move files to indicated test/train/data file
def write_files(rootpath, writepath, type_data, label, files):
    csv_path = os.path.join(writepath, type_data, type_data + '_labels')

    with open(csv_path, 'w') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames = ['image_id', 'label'])
        csv_writer.writeheader()

        for image in files:
            string_buffer = 9 - len(image)
            image_name = string_buffer * '0' + image

            csv_entry = {'image_id': image_name, 'label': label}
            csv_writer.writerow(csv_entry)

            cur_dir = os.path.join(rootpath, image)
            new_dir = os.path.join(writepath, type_data, 'images', image)
            shutil.move(cur_dir, new_dir)

# Identify image classes and create label file
def prep_data(data_path, write_path, type_data):
    label_class = os.listdir(data_path)
    for root, _, files in os.walk(data_path):
        label = check_file(root, label_class)
        write_files(root, write_path, type_data, label, files)

# Create train, test folders
def create_folder(path, split = False):
    if split:
        check_folder = ('train', 'test')
        for i in check_folder:
            folder_path = os.path.join(path, i)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
                os.makedirs(folder_path + '/images')
    else:
        folder_path = os.path.join(path, 'data')
        if not os.path.isdir(folder_path):       
            os.makedirs(folder_path)
            os.makedirs(folder_path + '/images')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument inputs
    parser.add_argument('--write-path', default='', help = 'Processed data write destination. PATH')
    parser.add_argument('--unroll-data', help = 'Unroll data into labels and images file. PATH')
    parser.add_argument('--train-path', help = 'Unroll train data into labels and images file. PATH')
    parser.add_argument('--test-path', help = 'Unroll test data into labels and images file. PATH')

    # Validate and assign argument inputs
    args = parser.parse_args()
    # check_args(args)

    write_path = args.write_path
    unroll_data = args.unroll_data
    train_path = args.train_path
    test_path = args.test_path

    # Helper vars
    split_bool = not bool(unroll_data)

    create_folder(write_path, split_bool)

    if test_path:
        prep_data(test_path, write_path,'test')

    if train_path:
        prep_data(train_path, write_path, 'train')

    if unroll_data:
        prep_data(unroll_data, write_path, 'data')