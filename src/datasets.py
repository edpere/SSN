import argparse
import glob
import json
import math
import os
import random
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


def prepare_dataset():
    random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration json file")
    parser.add_argument("folder", help="Folder with images")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    with open(args.config, "r") as cf:
        config = json.load(cf)

    mode = config["mode"]
    patch_size = config["patch_size"]
    batch_size = config["batch_size"]
    num_batches = config["num_batches"]

    num_patches = batch_size * num_batches

    if args.output:
        output_file = args.output + ".tfrec"
    else:
        output_file = "records.tfrec"

    files = glob.glob(os.path.join(args.folder, '*'))

    create_records(files,
                   mode=mode,
                   patch_size=patch_size,
                   max_patches=num_patches,
                   patch_stride=patch_size,
                   output=output_file,
                   method='1random')


def read_img(img_path, mode='grayscale', normalize=True):
    if mode == 'grayscale':
        img = Image.open(img_path).convert('L')
        img = np.expand_dims(np.asarray(img), -1)
    elif mode == 'rgb':
        img = Image.open(img_path)
        if img.mode != 'RGB':
            raise Exception('Trying to read {} image as RGB'.format(img.mode))
        img = np.asarray(img)
    else:
        raise Exception('Wrong mode. Got {}'.format(mode))

    if normalize:
        img = img / 255.0
        img = img.astype('float32')

    return img


def img_to_patches(img, patch_size, stride=1):
    """Convert an image to an array of square patches.

    Args:
        img: a numpy array representing the image
        patch_size: the size of the patches
        stride: the stride to be applied between patches

    Returns:
        numpy array of size (num_patches, patch_size, patch_size, channels) representing the patches
    """

    (img_h, img_w, img_c) = img.shape

    tmp = img[0:img_h - patch_size + 0 + 1:stride, 0:img_w - patch_size + 0 + 1:stride, :]
    num_patches = tmp.shape[0] * tmp.shape[1]

    patches = np.zeros([num_patches, patch_size * patch_size, img_c], dtype='float32')
    k = 0
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            tmp = img[i:img_h - patch_size + i + 1:stride, j:img_w - patch_size + j + 1:stride, :]
            patches[:, k, :] = tmp.reshape([num_patches, img_c])
            k += 1

    return patches.reshape([num_patches, patch_size, patch_size, img_c])


def extract_patch(img, patch_size, x=None, y=None):
    """Extract one patch from an image.

    Args:
        img: a numpy array representing the image
        patch_size: the size of the patch
        x: width coordinate of the top-left corner of the patch (if 'None' it's random)
        y: height coordinate of the top-left corner of the patch (if 'None' it's random)

    Returns:
        numpy array of size (patch_size, patch_size, channels) representing the patch
    """

    (img_h, img_w, img_c) = img.shape

    if y is None:
        y = np.random.randint(0, img_h - patch_size + 1)
    elif y > img_h - patch_size:
        raise Exception('y-dimension too large')

    if x is None:
        x = np.random.randint(0, img_w - patch_size + 1)
    elif x > img_w - patch_size:
        raise Exception('x-dimension too large')

    return img[y:y + patch_size, x:x + patch_size, :], x, y


# img.shape := (h,w,c)
def pad_image(img, scale):
    if img.shape[0] % scale != 0:
        rem = scale - img.shape[0] % scale
        img = np.concatenate([img, np.expand_dims(img[-rem, :, :], axis=0)], axis=0)
    if img.shape[1] % scale != 0:
        rem = scale - img.shape[1] % scale
        img = np.concatenate([img, np.expand_dims(img[:, -rem, :], axis=1)], axis=1)

    return img


def unpad_image(img, height, width):
    return img[:height, :width, :]


def print_progress(count_files, total_files, count_patches, max_patches):
    file_pct = count_files / total_files
    patch_pct = count_patches / max_patches

    msg = '\r Image progress: {0:.1%} images: {1:d} \t Patch progress: {2:.1%} patches: {3:d}'.format(file_pct,
                                                                                                      count_files,
                                                                                                      patch_pct,
                                                                                                      count_patches)
    sys.stdout.write(msg)
    sys.stdout.flush()


def create_records(files, mode='grayscale', patch_size=200, max_patches=2000000, patch_stride=200, output='records',
                   method='all'):
    print('Creating training dataset')
    print('patch size: %d' % patch_size)
    print('patch stride: %d' % patch_stride)
    print('max number of patches: %d' % max_patches)
    print('method: %s' % method)
    if method == 'nrandom':
        print('N: %d' % math.ceil(max_patches / len(files)))

    random.shuffle(files)

    with tf.io.TFRecordWriter(output) as writer:

        # Method 'all': select all the patches from an image until max_patches reached or no more images are available
        if method == 'all':
            curr_file = 0
            num_patches = 0

            while curr_file < len(files) and num_patches < max_patches:
                img = read_img(files[curr_file], mode=mode)
                patches = img_to_patches(img, patch_size=patch_size, stride=patch_stride)

                # Serialize each patch of this image
                for i in range(patches.shape[0]):
                    serialize_patch(writer, patches[i, :, :, :])
                    num_patches += 1

                curr_file += 1
                print_progress(curr_file, len(files), num_patches, max_patches)

        # Method 'nrandom': at each step randomly select n patch from 1 image
        if method == '1random' or method == 'nrandom':

            N = 1
            if method == 'nrandom':
                N = math.ceil(max_patches / len(files))

            curr_file = 0
            num_patches = 0

            while num_patches < max_patches:
                if curr_file >= len(files):
                    curr_file = 0

                img = read_img(files[curr_file], mode=mode)

                for _ in range(N):
                    patch, px, py = extract_patch(img, patch_size)
                    serialize_patch(writer, patch)

                    num_patches += 1

                curr_file += 1
                print_progress(curr_file, len(files), num_patches, max_patches)

    print('\nCreated %d patches from %d images' % (num_patches, curr_file))
    print('Output written in %s' % output)


def serialize_patch(writer, patch):
    patch_bytes = patch.tobytes()
    data = {'patch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_bytes]))}

    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())


if __name__ == "__main__":
    prepare_dataset()
