import argparse
import csv
import os
import shutil


# Check and return image label
def get_label(filepath):
    label = filepath.split('/')[-1]
    return label

# Move files to indicated test/train/data file
def write_files(rootpath, writepath, label, files):
    csv_path = os.path.join(writepath, 'data_labels.csv')

    with open(csv_path, 'a') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames = ['image_id', 'label'])

        for image in files:
            csv_entry = {'image_id': image, 'label': label}
            csv_writer.writerow(csv_entry)

            cur_dir = os.path.join(rootpath, image)
            new_dir = os.path.join(writepath, 'images', image)
            shutil.move(cur_dir, new_dir)

# Identify image classes and create label file
def prep_data(data_path, write_path, type_data):

    for root, _, files in os.walk(data_path):
        label = get_label(root)
        write_files(root, write_path, label, files)

    print("Finished copying files.")

# Create train, test folders
def create_folder(path):
    if not os.path.isdir(path):       
        os.makedirs(path)
        os.makedirs(path + '/images')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument inputs
    parser.add_argument('--write', default='', help = 'Processed data write destination. PATH')
    parser.add_argument('--unroll', required = True, help = 'Unroll data into labels and images file. PATH')

    # Validate and assign argument inputs
    args = parser.parse_args()

    write_path = args.write
    unroll_path = args.unroll

    create_folder(write_path)

    # Unroll data
    if unroll_path:
        prep_data(unroll_path, write_path, 'data')