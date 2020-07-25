import argparse
import csv
import os
import shutil


def check_file(filepath):
    label_class = ('forest', 'buildings', 'glacier', 'street', 'mountain', 'sea')
    label = filepath.split('/')[-1]

    if label in label_class:
        return label
    else:
        return False

def write_files(rootpath, writepath, filename, label, files):
    with open(os.path.join(writepath, filename), 'w') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames = ['image_id', 'label'])
        csv_writer.writeheader()

        for image in files:
            csv_entry = dict()
            string_buffer = 9 - len(image)
            csv_entry['image_id'] = string_buffer * '0' + image
            csv_entry['label'] = label
            csv_writer.writerow(csv_entry)

            cur_dir = os.path.join(rootpath, image)
            new_dir = os.path.join(writepath, 'images', image)
            shutil.move(cur_dir, new_dir)

def prep_pred(path = './data/111880_269359_bundle_archive/seg_pred/seg_pred'):
    write_path = './data/pred'

    for root, _, files in os.walk(path):
            label = check_file(root)
            if not label:
                continue

            for image in files:
                cur_dir = os.path.join(root, image)
                new_dir = os.path.join(write_path, 'image', image)
                shutil.move(cur_dir, new_dir)

def prep_test(path = './data/111880_269359_bundle_archive/seg_test/seg_test'):
    write_path = './data/test'

    for root, _, files in os.walk(path):
        label = check_file(root)
        if not label:
            continue

        write_files(root, write_path, 'test_labels', label, files)

def prep_train(path = './data/111880_269359_bundle_archive/seg_train/seg_train'):
    write_path = './data/train'

    for root, _, files in os.walk(path):
        label = check_file(root)
        if not label:
            continue

        write_files(root, write_path, 'train_labels', label, files)

# Create train, test, pred folders
def create_folder():
    data_path = './data'
    check_folder = ('train', 'test', 'pred')
    for i in check_folder:
        folder_path = os.path.join(data_path, i)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            os.mkdir(os.path.join(folder_path, 'images'))


if __name__ == '__main__':
    
    create_folder()
    prep_test()