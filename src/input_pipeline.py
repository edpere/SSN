import glob
import os

import tensorflow as tf

from datasets import read_img, pad_image


# Parse an example serialized in a TFRecord
def parse_example(serialized):
    feature = {'patch': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(serialized=serialized, features=feature)
    patch = example['patch']
    patch = tf.io.decode_raw(patch, tf.float32)

    return patch


def data_augmentation(patch):
    patch = tf.image.random_flip_left_right(patch)
    patch = tf.image.random_flip_up_down(patch)
    patch = tf.image.rot90(patch, tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32))

    return patch


def sample_sigma(sigma_fix):
    if sigma_fix is None:
        sigma = tf.random.uniform(shape=(), minval=0, maxval=76, dtype='int32')
    elif len(sigma_fix) == 2:
        sigma = tf.random.uniform(shape=(), minval=sigma_fix[0], maxval=sigma_fix[1], dtype='int32')
    else:
        sigma = sigma_fix[0]

    sigma = tf.cast(sigma, 'float32')

    return sigma


def add_noise_map(patch, sigma_fix=None, maps_c=1):
    sigma = sample_sigma(sigma_fix)

    noise = tf.random.normal(tf.shape(patch), mean=0, stddev=sigma, dtype='float32')
    noise = noise / 255.0

    noisy_patch = tf.add(patch, noise)
    noise_map = tf.fill((tf.shape(patch)[0], tf.shape(patch)[1], maps_c), sigma)
    noise_map = noise_map / 255.0

    return (noisy_patch, noise_map), patch


def add_noise(patch, sigma_fix=None):
    sigma = sample_sigma(sigma_fix)

    noise = tf.random.normal(tf.shape(patch), mean=0, stddev=sigma, dtype='float32')
    noise = noise / 255.0

    noisy_patch = tf.add(patch, noise)

    return noisy_patch, patch


# Create a dataset object from TFRecords
def create_dataset(file_name='records', batch_size=2, patch_size=200, channels=1, sigma_fix=None, noise_map=False,
                   augmentation=True):
    dataset = tf.data.TFRecordDataset(filenames=file_name)
    dataset = dataset.map(parse_example)
    dataset = dataset.map(lambda x: tf.reshape(x, [patch_size, patch_size, channels]))

    if augmentation:
        dataset = dataset.map(data_augmentation)

    if noise_map:
        dataset = dataset.map(lambda x: add_noise_map(x, sigma_fix))
    else:
        dataset = dataset.map(lambda x: add_noise(x, sigma_fix))

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    return dataset


def valid_generator(path, mode='grayscale', batch_size=1, sigma_fix=None, noise_map_input=True, maps_c=1, scale=1):
    files = glob.glob(os.path.join(path, '*'))

    for f in files:

        img = read_img(f, mode)

        # Check dimensions and, if necessary, pad the image
        img = pad_image(img, scale)

        img = tf.expand_dims(img, axis=0)

        sigma = sample_sigma(sigma_fix)

        noise = tf.random.normal(img.shape, mean=0, stddev=sigma, dtype='float32')
        noise = noise / 255.0

        noisy_img = tf.add(img, noise)

        if noise_map_input:
            noise_map = tf.fill([1, img.shape[1], img.shape[2], maps_c], sigma / 255.0)
            yield (noisy_img, noise_map), img
        else:
            yield noisy_img, img


def get_valid_dataset(valid_path, mode, sigma_fix, noise_map_input=False, scale=1):
    if mode == 'grayscale':
        channels = 1
    elif mode == 'RGB':
        channels = 3
    else:
        raise Exception('Wrong mode')

    if noise_map_input:
        valid_dataset = tf.data.Dataset.from_generator(lambda: valid_generator(path=valid_path,
                                                                               mode=mode,
                                                                               sigma_fix=sigma_fix,
                                                                               noise_map_input=noise_map_input,
                                                                               scale=scale),
                                                       output_types=(('float32', 'float32'), 'float32'),
                                                       output_shapes=(
                                                       ([None, None, None, channels], [None, None, None, channels]),
                                                       [None, None, None, channels])
                                                       )
    else:
        valid_dataset = tf.data.Dataset.from_generator(lambda: valid_generator(path=valid_path,
                                                                               mode=mode,
                                                                               sigma_fix=sigma_fix,
                                                                               noise_map_input=noise_map_input,
                                                                               scale=scale),
                                                       output_types=('float32', 'float32'),
                                                       output_shapes=(
                                                       [None, None, None, channels], [None, None, None, channels])
                                                       )

    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    len_valid = len(glob.glob(os.path.join(valid_path, '*')))

    return valid_dataset, len_valid
