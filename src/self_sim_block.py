import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Shift(tf.keras.layers.Layer):
    def __init__(self, vx, vy, mode='constant', name=None):
        super(Shift, self).__init__(name=name)
        self.vx = vx
        self.vy = vy
        self.mode = mode
        self.dx1 = self.dx2 = self.dy1 = self.dy2 = 0

        if vx > 0:  # shift right
            self.dx1 = vx
        elif vx < 0:  # shift left
            self.dx2 = -vx

        if vy > 0:  # shift down
            self.dy1 = vy
        elif vy < 0:  # shift up
            self.dy2 = -vy

    def call(self, inputs):
        t = inputs

        if self.vx > 0:
            t = t[:, :, :-self.vx, :]
        elif self.vx < 0:
            t = t[:, :, -self.vx:, :]

        if self.vy > 0:
            t = t[:, :-self.vy, :, :]
        elif self.vy < 0:
            t = t[:, -self.vy:, :, :]

        pad = [[0, 0], [self.dy1, self.dy2], [self.dx1, self.dx2], [0, 0]]
        pt = tf.pad(t, pad, mode=self.mode, constant_values=0)

        return pt


class PatchFolding(tf.keras.layers.Layer):
    def __init__(self, patch_size, channels, name=None):
        super(PatchFolding, self).__init__(name=name)

        kernel = np.eye(patch_size * patch_size * channels)
        kernel = kernel.reshape(patch_size, patch_size, channels, patch_size * patch_size * channels)

        self.kernel = tf.Variable(initial_value=tf.convert_to_tensor(kernel, dtype=tf.float32), trainable=False)

    def call(self, inputs):
        return tf.nn.conv2d(inputs,
                            self.kernel,
                            strides=[1, 1, 1, 1],
                            padding='SAME')


def get_self_sim_block_residual(input_feature, id, nfeature=64, psz=3, hnsz=4, stride=2, Norm=None, NormParams=[],
                                weight_decay=0, weights="softmax", patch_folding_size=1, shift_pad="constant",
                                verbose=False):
    if verbose:
        print("-" * 30)

    input_channels = input_feature[0].shape[-1]
    input_tensor = tf.keras.Input(shape=input_feature[0].shape)

    if patch_folding_size > 1:
        input_feature = PatchFolding(patch_folding_size, input_channels)(input_tensor)
    else:
        input_feature = input_tensor

    feature_shared = tf.keras.layers.Convolution2D(filters=nfeature, kernel_size=(1, 1), padding='same',
                                                   kernel_initializer='he_normal',
                                                   kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                                                   name='ref_conv')(input_feature)
    feature_neigh = tf.keras.layers.Convolution2D(filters=nfeature, kernel_size=(1, 1), padding='same',
                                                  kernel_initializer='he_normal',
                                                  kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                                                  name='neigh_conv')(input_feature)

    all_shift_neigh = []
    all_shift_input = []

    if verbose:
        print("Type of shifts: {}".format(shift_pad))

    for i in range(-hnsz, hnsz + 1, stride):
        for j in range(-hnsz, hnsz + 1, stride):
            if verbose:
                print("Shift({0},{1})".format(i, j))

            name = ''
            if i < 0:
                name += 'n' + str(abs(i))
            else:
                name += 'p' + str(i)
            if j < 0:
                name += 'n' + str(abs(j))
            else:
                name += 'p' + str(j)

            if shift_pad == 'roll':
                all_shift_neigh.append(
                    tf.roll(feature_neigh, shift=[i, j], axis=[1, 2]))  # axis 1,2 because the 0-th is the batch axis
                all_shift_input.append(
                    tf.roll(input_feature, shift=[i, j], axis=[1, 2]))  # axis 1,2 because the 0-th is the batch axis
            elif shift_pad == 'constant' or shift_pad == 'reflect' or shift_pad == 'symmetric':
                name = 'Shift' + name

                all_shift_neigh.append(Shift(i, j, mode=shift_pad, name=name + '-f')(feature_neigh))
                all_shift_input.append(Shift(i, j, mode=shift_pad, name=name + '-i')(input_feature))
            else:
                raise Exception('Wrong shift type')

    if verbose:
        print("Number of shifts: {0}".format(len(all_shift_neigh)))

    shared_conv = tf.keras.layers.Convolution2D(filters=nfeature, kernel_size=(1, 1), padding='same',
                                                kernel_initializer='he_normal',
                                                kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                                                name='local_pred')

    all_pred = []
    all_dst_tensor = []
    for i in range(len(all_shift_neigh)):

        t = tf.keras.layers.add([feature_shared, all_shift_neigh[i]])

        if Norm == 'Bnorm':
            t = tf.keras.layers.BatchNormalization(momentum=NormParams[0])(t)
            # t = feature_bnorm(t)
        elif Norm == 'Gnorm':
            t = tfa.layers.GroupNormalization(groups=NormParams[0])(t)

        dst_tensor = tf.keras.layers.ReLU()(t)

        concat = tf.keras.layers.Concatenate()([dst_tensor, all_shift_input[i]])

        t = shared_conv(concat)

        if Norm == 'Bnorm':
            t = tf.keras.layers.BatchNormalization(momentum=NormParams[0])(t)
            # t = pred_bnorm(t)
        elif Norm == 'Gnorm':
            t = tfa.layers.GroupNormalization(groups=NormParams[0])(t)

        local_pred = tf.keras.layers.ReLU()(t)

        all_dst_tensor.append(dst_tensor)
        all_pred.append(local_pred)

    # Compute weights
    if weights == "softmax":
        if verbose:
            print("Softmax weights")
        all_weights_matrix = get_weights_softmax(all_dst_tensor)
    else:
        if verbose:
            print("Linear weights")
        all_weights_matrix = get_weights_linear(all_dst_tensor)

    # Weighted prediction
    all_weighted_pred = []
    for i in range(len(all_pred)):
        all_weighted_pred.append(tf.math.multiply(all_pred[i], all_weights_matrix[i]))

    final_pred = tf.keras.layers.add(all_weighted_pred)

    final_pred = tf.keras.layers.Convolution2D(filters=input_channels, kernel_size=(1, 1), padding='same',
                                               name='residual_conv', kernel_initializer='zeros',
                                               kernel_regularizer=tf.keras.regularizers.L2(weight_decay))(
        final_pred)  # Init to 0

    # Residual connection
    output_feature = tf.keras.layers.add([final_pred, input_tensor])

    return tf.keras.Model(inputs=input_tensor, outputs=output_feature, name=id)


def get_weights_linear(all_dst_tensor):
    reduce_conv = tf.keras.layers.Convolution2D(filters=1, kernel_size=(1, 1), padding='same',
                                                kernel_initializer='he_normal',
                                                name='reduce_channel')  # channel reduction

    all_dst_matrix = [reduce_conv(x) for x in all_dst_tensor]

    norm_matrix = tf.keras.layers.add(all_dst_matrix, name="Norm_Matrix")  # Normalization matrix

    all_weights_matrix = [tf.math.divide_no_nan(x, norm_matrix) for x in all_dst_matrix]

    return all_weights_matrix


def get_weights_softmax(all_dst_tensor, weight_decay=0):
    reduce_conv = tf.keras.layers.Convolution2D(filters=1, kernel_size=(1, 1), padding='same',
                                                kernel_initializer='he_normal',
                                                kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
                                                name='reduce_channel')  # channel reduction

    all_dst_matrix = [reduce_conv(x) for x in all_dst_tensor]

    dst_volume = tf.concat(all_dst_matrix, axis=-1)

    softmax_weights = tf.nn.softmax(dst_volume, -1)

    all_weights_matrix = []

    for i in range(softmax_weights.shape[-1]):
        all_weights_matrix.append(softmax_weights[:, :, :, i:i + 1])

    return all_weights_matrix
