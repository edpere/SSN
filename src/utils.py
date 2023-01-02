import datetime
import os

import numpy as np
import tensorflow as tf


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, tf.clip_by_value(y_pred, 0, 1), 1)


def get_lr_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)


# Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch > 1 and epoch % 20 == 0:  # 20
        return lr / 10
    if epoch > 21 and epoch % 10 == 0:
        return lr / 10
    else:
        return lr


# Checkpoints
def get_ckpt_scheduler(base_dir='ckpt', freq='epoch', model_name=None):
    os.makedirs(base_dir, exist_ok=True)

    if model_name is None:
        ckpt_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        ckpt_dir = model_name

    os.mkdir(os.path.join(base_dir, ckpt_dir))

    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(base_dir, ckpt_dir, 'weights-{epoch:02d}.h5'),
                                              save_weights_only=True,
                                              save_freq=freq)


def get_tensorboard_cb(base_dir='logs', profile_batch='500,505', model_name=None):
    os.makedirs(base_dir, exist_ok=True)

    if model_name is None:
        logs_dir = os.path.join(base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        logs_dir = os.path.join(base_dir, model_name)

    return tf.keras.callbacks.TensorBoard(logs_dir, profile_batch=profile_batch)


# Callback functions for svd orthogonalization of convolutional layers
class Orthogonalizer(tf.keras.callbacks.Callback):
    def __init__(self, period=10):
        super(Orthogonalizer, self).__init__()

        self.period = period

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.period != 0:
            return

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                weights = layer.get_weights()[0]
                (k1, k2, c_in, c_out) = weights.shape
                weights = np.reshape(weights, [k1 * k2 * c_in, c_out])
                (u, _, v) = np.linalg.svd(weights, full_matrices=False)
                weights = np.dot(u, v)
                layer.set_weights = np.reshape(weights, [k1, k2, c_in, c_out])


def get_svd_regularizer(period):
    return Orthogonalizer(period)


def get_confirmation(msg):
    x = ''
    while x != 'y' and x != 'n':
        print(msg + ' [y/n]')

        x = input().lower()
        if x == 'n':
            exit(0)


def get_model_name(name_conv, config, weights=None):
    if name_conv is None or name_conv == 'date':
        model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    elif name_conv == 'model':
        model_name = config["general"]["base_model"]
        model_name += '_' + config["general"]["mode"]

        if weights is not None:
            model_name += '_' + weights.split('/')[-1]

        if not config["model"]["block"] is None:
            model_name += '_' + config["model"]["block"]
            model_name += '_' + config["SelfSimBlocks"]["shift_pad"]
            if config["SelfSimBlocks"]["patch_folding_size"] > 1:
                model_name += '_pfs' + str(config["SelfSimBlocks"]["patch_folding_size"])

        model_name += '_sigma' + list_to_str(config["general"]["sigma"])
        model_name += '_d' + str(config["model"]["depth"])
        model_name += '_nf' + str(config["model"]["num_filters"])

        if not config["model"]["block"] is None:
            model_name += '_pos' + list_to_str(config["SelfSimBlocks"]["block_pos"])
            model_name += '_h' + list_to_str(config["SelfSimBlocks"]["hnsz"])
            model_name += '_s' + list_to_str(config["SelfSimBlocks"]["stride"])
            model_name += '_bwd' + str(config["SelfSimBlocks"]["weight_decay"])
            if config["SelfSimBlocks"]["Norm"]:
                model_name += '_b'
                if config["SelfSimBlocks"]["Norm"] == 'Bnorm':
                    model_name += 'BN'
                if config["SelfSimBlocks"]["Norm"] == 'Gnorm':
                    model_name += 'GN'
                model_name += list_to_str(config["SelfSimBlocks"]["NormParams"])

            model_name += '_' + config["SelfSimBlocks"]["block_model"]
        model_name += '_wd' + str(config["model"]["weight_decay"])

        if config["model"]["Norm"]:
            model_name += '_'
            if config["model"]["Norm"] == 'Bnorm':
                model_name += 'BN'
            if config["model"]["Norm"] == 'Gnorm':
                model_name += 'GN'
            model_name += list_to_str(config["SelfSimBlocks"]["NormParams"])

        model_name += '_lr' + str(config["training"]["init_lr"])
        model_name += '_loss' + config["training"]["loss"]

        if not config["training"]["other"] is None:
            model_name += '_' + list_to_str(config["training"]["other"])

    return model_name


def list_to_str(l):
    s = '['
    if len(l) > 0:
        s += str(l[0])
        for i in range(1, len(l)):
            s += '_' + str(l[i])
    s += ']'

    return s
