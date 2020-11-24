import copy
import skimage.measure
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import os
import pandas as pd
from tqdm import tqdm

def convert_binary_to_img(fpath, verbose=False):
    """
    convert binary files to images (255*255*1), to select the best fraction of texture, use the Shanno entropy
    :param fpath: file path for the binary files
    :param verbose: if true, show the image
    :return: image in 2d ndarray
    """
    with open(fpath, 'r') as f:
        pixels = []
        for line in f.readlines():
            line = line.strip('\n')
            try:
                pixels.extend([int(item, 16) for item in line.split(' ')[1:]])
            except:
                pass
    num_lines = len(pixels) // 256
    pixels = pixels[: num_lines * 256]
    pixel_array = np.array(pixels, dtype=np.uint8).reshape((num_lines, 256))
    entropy_list = []

    if pixel_array.shape[0] < 256:
        padding = np.zeros((256, 256), dtype=np.uint8)
        padding[:pixel_array.shape[0], :] = pixel_array
        pixel_array = padding

    for i in range(0, pixel_array.shape[0] - 255, 20):
        entropy = skimage.measure.shannon_entropy(pixel_array[i:i + 256, :])
        entropy_list.append((entropy, i))

    def sort_key(item):
        return -item[0]

    entropy_list = sorted(entropy_list, key=sort_key)
    try:
        extrated_texture = pixel_array[entropy_list[0][1]:entropy_list[0][1] + 256, :]
    except:
        pass
    if verbose:
        plt.imshow(extrated_texture, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return extrated_texture

def create_img_dataset(original_foler, train_img_folder, test_img_folder, csv_file, split=0.2):
    """
    Creating image dataset
    :param original_foler: the directory holding binary files
    :param train_img_folder: the directroy for storing training images
    :param test_img_folder: the directroy for storing training images
    :param csv_file: the training csv label file
    :param split: train and test split
    :return: None
    """
    df = pd.read_csv(csv_file)
    df = df.iloc[np.random.permutation(len(df))]
    num_rows = len(df)
    train_count = num_rows * (1. - split)
    if not os.path.exists(train_img_folder):
        os.mkdir(train_img_folder)
    if not os.path.exists(test_img_folder):
        os.mkdir(test_img_folder)

    for [id, cat] in tqdm(df.values):
        original_file = os.path.join(original_foler, id + '.bytes')
        if train_count > 0:
            new_folder = train_img_folder
            train_count -= 1
        else:
            new_folder = test_img_folder
        target_cat_folder = os.path.join(new_folder, str(cat))
        if not os.path.exists(target_cat_folder):
            os.mkdir(target_cat_folder)
        target_file = os.path.join(new_folder, str(cat), id+'.jpg')
        assert os.path.exists(original_file)
        extrated_texture = convert_binary_to_img(original_file)
        imsave(target_file, extrated_texture)

if __name__ == '__main__':
    training_folder = "/home/junzheng/course/CSI5137/csi5137Project/malware-classification/train"
    train_img_folder = "/home/junzheng/course/CSI5137/csi5137Project/malware-classification/train_imgs"
    test_img_folder = "/home/junzheng/course/CSI5137/csi5137Project/malware-classification/test_imgs"
    csv_file = "/home/junzheng/course/CSI5137/csi5137Project/malware-classification/trainLabels.csv"
    create_img_dataset(training_folder, train_img_folder, test_img_folder, csv_file)