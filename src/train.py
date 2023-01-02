import argparse
import json
import pickle

import tensorflow as tf

from ffdnet import load_weights_mat
from input_pipeline import create_dataset, get_valid_dataset
from models import get_model
from utils import get_model_name, get_confirmation, psnr_metric, get_lr_scheduler, get_ckpt_scheduler, \
    get_svd_regularizer

parser = argparse.ArgumentParser(description="Training of SSN")

parser.add_argument("config", help="Configuration json file")
parser.add_argument("training_set", help="tfrecord of the training dataset")
parser.add_argument("validation_set", nargs='?', help="Directory with validation images")
parser.add_argument("-w", "--weights", help="Weights for initializing the model")
parser.add_argument("-v", "--verbosity", type=int)

args = parser.parse_args()

# print(args)

verbosity = args.verbosity
if verbosity is None:
    verbosity = 0
# print(verbosity)

with open(args.config, "r") as cf:
    config = json.load(cf)

# print(config)

base_model = config["general"]["base_model"]
name_conv = config["general"]["name_conv"]
mode = config["general"]["mode"]
patch_size = config["general"]["patch_size"]
batch_size = config["general"]["batch_size"]
num_batches = config["general"]["num_batches"]
sigma = config["general"]["sigma"]
augmentation = config["general"]["augmentation"]

residual_learning = config["model"]["residual_learning"]
sub_residual_learning = config["model"]["sub_residual_learning"]
block = config["model"]["block"]
depth = config["model"]["depth"]
num_filters = config["model"]["num_filters"]
kernel_size = config["model"]["kernel_size"]
weight_decay = config["model"]["weight_decay"]
Norm = config["model"]["Norm"]
NormParams = config["model"]["NormParams"]
scale = config["model"]["scale"]

shift_pad = config["SelfSimBlocks"]["shift_pad"]
block_pos = config["SelfSimBlocks"]["block_pos"]
block_model = config["SelfSimBlocks"]["block_model"]
hnsz = config["SelfSimBlocks"]["hnsz"]
stride = config["SelfSimBlocks"]["stride"]
block_weight_decay = config["SelfSimBlocks"]["weight_decay"]
block_norm = config["SelfSimBlocks"]["Norm"]
block_normparams = config["SelfSimBlocks"]["NormParams"]
patch_folding_size = config["SelfSimBlocks"]["patch_folding_size"]

if block is not None:
    SelfSimBlocks = config["SelfSimBlocks"]
else:
    SelfSimBlocks = None

if block_weight_decay < 0:
    block_weight_decay = weight_decay
    config["SelfSimBlocks"]["weight_decay"] = weight_decay

init_lr = config["training"]["init_lr"]
epochs = config["training"]["epochs"]
loss = config["training"]["loss"]
ckpt_freq_epochs = config["training"]["ckpt_freq_epochs"]

if verbosity >= 1:
    print("\nGENERAL SETTINGS")
    print("base_model: {}".format(base_model))
    print("name_conv: {}".format(name_conv))
    print("mode: {}".format(mode))
    print("patch_size: {}".format(patch_size))
    print("batch_size: {}".format(batch_size))
    print("num_batches: {}".format(num_batches))
    print("sigma: {}".format(sigma))
    print("augmentation: {}".format(augmentation))

    print("\nMODEL HYPERPARAMETERS")
    print("residual_learning: {}".format(residual_learning))
    print("sub_residual_learning: {}".format(sub_residual_learning))
    print("block: {}".format(block))
    print("depth: {}".format(depth))
    print("num_filters: {}".format(num_filters))
    print("kernel_size: {}".format(kernel_size))
    print("weight_decay: {}".format(weight_decay))
    print("Norm: {}".format(Norm))
    print("NormParams: {}".format(NormParams))
    print("scale: {}".format(scale))

    print("\nSELFSIMBLOCK SETTINGS")
    print("shift_pad: {}".format(shift_pad))
    print("block_pos: {}".format(block_pos))
    print("block_model: {}".format(block_model))
    print("hnsz: {}".format(hnsz))
    print("stride: {}".format(stride))
    print("block_weight_decay: {}".format(block_weight_decay))
    print("block_norm: {}".format(block_norm))
    print("block_normparams: {}".format(block_normparams))
    print("patch_folding_size: {}".format(patch_folding_size))

    print("\nTRAINING SETTINGS")
    print("init_lr: {}".format(init_lr))
    print("epochs: {}".format(epochs))
    print("loss: {}".format(loss))
    print("ckpt_freq_epochs: {}".format(ckpt_freq_epochs))
    print("other: {}".format(config["training"]["other"]))

model_name = get_model_name(name_conv, config, args.weights)
print('\nModel name: {}'.format(model_name))
print('Weights: {}'.format(args.weights))
print("Training set: {}".format(args.training_set))
print("Validation set: {}".format(args.validation_set))

device_name = tf.test.gpu_device_name()
if not device_name:
    print('GPU device not found')
else:
    print('Found GPU at: {}'.format(device_name))

get_confirmation('Continue?')

# CREATE MODEL

print('\nCreating model')

model = get_model(base_model=base_model,
                  mode=mode,
                  residual_learning=residual_learning,
                  sub_residual_learning=sub_residual_learning,
                  num_filters=num_filters,
                  depth=depth,
                  kernel_size=kernel_size,
                  weight_decay=weight_decay,
                  Norm=Norm,
                  NormParams=NormParams,
                  scale=scale,
                  SelfSimBlocks=SelfSimBlocks,
                  verbose=verbosity)

if verbosity >= 1:
    model.summary()

# LOAD WEIGHTS

if args.weights:
    print('\nLoading weights from: {}'.format(args.weights))
    if base_model == 'ffdnet':
        load_weights_mat(args.weights, model.layers[3], verbose=verbosity)
    else:
        load_weights_mat(args.weights, model, verbose=verbosity)

# PREPARE DATASETS

print('\nPreparing datasets')

if mode == 'grayscale':
    image_channels = 1
elif mode.lower() == 'rgb':
    image_channels = 3
else:
    raise Exception('Invalid mode for input images. Got: ' + str(mode))

if base_model == 'ffdnet':
    noise_map = True
else:
    noise_map = False

train_dataset = create_dataset(file_name=args.training_set,
                               batch_size=batch_size,
                               patch_size=patch_size,
                               channels=image_channels,
                               sigma_fix=sigma,
                               noise_map=noise_map)

if args.validation_set:
    valid_dataset, len_valid = get_valid_dataset(valid_path=args.validation_set,
                                                 mode=mode,
                                                 sigma_fix=sigma,
                                                 noise_map_input=noise_map,
                                                 scale=scale)
else:
    valid_dataset = None
    len_valid = 0

# COMPILE MODEL

print('\nCompiling model')

if loss.lower() == 'l2':
    loss_func = tf.keras.losses.MeanSquaredError()
elif loss.lower() == 'l1':
    loss_func = tf.keras.losses.MeanAbsoluteError()

optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)

model.compile(optimizer=optimizer,
              loss=loss_func,
              metrics=[psnr_metric]
              )

# TRAINING CALLBACKS

print('\nPreparing callbacks')

if verbosity >= 1:
    print("-ckpt frequency: {0} epochs ({1} batches)".format(ckpt_freq_epochs, ckpt_freq_epochs * num_batches))

callbacks = []

callbacks.append(get_lr_scheduler())
callbacks.append(get_ckpt_scheduler(freq=ckpt_freq_epochs * num_batches, model_name=model_name))

if not config["training"]["other"] is None and "svd_reg" in config["training"]["other"]:
    callbacks.append(get_svd_regularizer(16))

print(callbacks)


# TRAINING

get_confirmation("Start training?")

print('OK :)')

history = model.fit(x=train_dataset,
                    validation_data=valid_dataset,
                    epochs=epochs,
                    steps_per_epoch=num_batches,
                    validation_steps=len_valid,
                    callbacks=callbacks
                    )


# SAVE RESULTS

with open('history_' + model_name, 'wb') as f:
    pickle.dump(history.history, f)

model.save_weights(model_name + '.h5')
