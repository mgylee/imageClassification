import argparse
import csv
import os
import shutil


def check_args(args):
    assert args.unroll_data and not (args.test_path or args.train_path), "Provide --unroll-data only or --train-path and --test-path."

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
            string_buffer = 9 - len(image)
            image_name = string_buffer * '0' + image

            csv_entry = {'image_id': image_name, 'label': label}
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
def create_folder(path, split = False):
    if split:
        check_folder = ('train', 'test')
        for i in check_folder:
            folder_path = os.path.join(path, i)
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
                os.mkdir(folder_path + '/images')
    else:
        folder_path = os.path.join(path, 'data')
        if not os.path.isdir(folder_path):       
            os.mkdir(folder_path)
            os.mkdir(folder_path + '/images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument inputs
    parser.add_argument('--write-path', default='', help = 'Processed data write destination. PATH')
    parser.add_argument('--unroll-data', help = 'Unroll data into labels and images file. PATH')
    parser.add_argument('--train-path', help = 'Unroll train data into labels and images file. PATH')
    parser.add_argument('--test-path', help = 'Unroll test data into labels and images file. PATH')

    # Validate and assign argument inputs
    args = parser.parse_args()
    check_args(args)

    write_path = args.write_path
    unroll_data = args.unroll_data
    train_path = args.train_path
    test_path = args.test_path
    

    create_folder()
    #prep_test()