import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import scipy.io as sio
import h5py

from self_sim_block import get_self_sim_block_residual


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale=2):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    # inputs is [imgs: (bs,h,w,c_i), noise maps: (bs,h,w,c_m)]
    def call(self, inputs):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PixelShuffle must be called on a list of 2 tensors (batch of images and noise lavel maps). Got: ' + str(
                    inputs))

        images = inputs[0]
        maps = inputs[1]

        if len(images.shape) != 4 or len(maps.shape) != 4:
            raise Exception(
                'PixelShuffle received wrong number of dimensions for images and/or maps (expect 4). Got: images.dim=' + len(
                    images.shape) != 4
                + ' maps.dim=' + len(maps.shape))

        (_, _, _, c) = images.shape
        bs = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]

        # scale of the subsampling
        scale = self.scale
        scale2 = scale * scale

        # output tensor dimensions 
        h_out = h // scale
        w_out = w // scale
        c_out = c * scale2

        stack = tf.zeros([bs, h_out, w_out, 0])
        for channel in range(c):
            for si in range(scale2):
                a = np.mod(si, scale)
                b = int(np.floor(si / scale))

                sub_img = images[:, a::scale, b::scale, channel]
                sub_img = tf.expand_dims(sub_img, -1)

                stack = tf.concat([stack, sub_img], axis=-1)

        noise_maps = tf.nn.avg_pool2d(input=maps, ksize=scale, strides=scale, padding='VALID')

        stack = tf.concat([stack, noise_maps], axis=-1)

        return stack


class PixelUnShuffle(tf.keras.layers.Layer):
    def __init__(self, scale=2):
        super(PixelUnShuffle, self).__init__()
        self.scale = scale

    # inputs shape is (bs, h/scale, w/scale, c*scale*scale)
    def call(self, inputs):
        scale = self.scale
        scale2 = scale * scale

        (_, _, _, c) = inputs.shape
        bs = tf.shape(inputs)[0]
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]

        if c % scale2 != 0:
            raise Exception('PixelUnShuffle received uncompatible number of channels for the given scale. Got ' + str(
                c) + ' channels but scale is ' + str(scale))

        h_out = h * scale
        w_out = w * scale
        c_out = c // scale2

        output = tf.zeros([bs, h_out, w_out, 0])

        # for each original channel perform unshuffling
        for channel in range(c_out):
            sub_imgs = []
            for si in range(scale2):
                sub_imgs.append(inputs[:, :, :, scale2 * channel + si])

            stacked_list = []
            for j in range(scale):
                stacked_list.append(tf.stack(sub_imgs[j::scale], axis=3))

            res = tf.stack(stacked_list, axis=2)
            res = tf.reshape(res, [bs, h_out, w_out, 1])

            output = tf.concat([output, res], axis=-1)

        return output


def dncnn(mode='grayscale', residual_learning=True, num_filters=None, depth=None, input_channels=None,
          output_channels=None, kernel_size=3, weight_decay=0, Norm='Bnorm', NormParams=[], SelfSimBlocks=None, id=''):
    num_filters = 64 if num_filters is None else num_filters
    depth = 17 if depth is None else depth

    if mode == 'grayscale':
        image_channels = 1
    elif mode == 'RGB':
        image_channels = 3
    else:
        raise Exception('Invalid mode for input images. Got: ' + str(mode))

    input_channels = image_channels if input_channels is None else input_channels
    output_channels = input_channels if output_channels is None else output_channels

    img_input = tf.keras.Input(shape=[None, None, input_channels])

    layer = tf.keras.layers.Conv2D(filters=num_filters,
                                   kernel_size=kernel_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   bias_initializer='zeros',
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
                                   )(img_input)
    layer = tf.keras.layers.ReLU()(layer)

    block_id = 0  # Id of SelfSim Blocks
    for i in range(1, depth - 1):

        # Insert new SelfSim Block
        if not (SelfSimBlocks is None) and i in SelfSimBlocks['block_pos']:
            # Get proper hnsz
            if block_id >= len(SelfSimBlocks['hnsz']):
                next_hnsz = SelfSimBlocks['hnsz'][-1]
            else:
                next_hnsz = SelfSimBlocks['hnsz'][block_id]

            # Ger proper stride
            if block_id >= len(SelfSimBlocks['stride']):
                next_stride = SelfSimBlocks['stride'][-1]
            else:
                next_stride = SelfSimBlocks['stride'][block_id]

            layer = get_self_sim_block_residual(layer, id='SelfSimBlock' + str(block_id),
                                                hnsz=next_hnsz, stride=next_stride,
                                                Norm=SelfSimBlocks['Norm'],
                                                NormParams=SelfSimBlocks['NormParams'],
                                                weight_decay=SelfSimBlocks["weight_decay"],
                                                patch_folding_size=SelfSimBlocks["patch_folding_size"],
                                                shift_pad=SelfSimBlocks['shift_pad'],
                                                verbose=2)(layer)

            block_id += 1

        layer = tf.keras.layers.Conv2D(filters=num_filters,
                                       kernel_size=kernel_size,
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
                                       )(layer)
        if Norm == 'Bnorm':
            layer = tf.keras.layers.BatchNormalization(momentum=NormParams[0])(layer)
        elif Norm == 'Gnorm':
            layer = tfa.layers.GroupNormalization(groups=NormParams[0])(layer)

        layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(  # filters=image_channels*scale*scale,
        filters=output_channels,
        kernel_size=kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(layer)

    if residual_learning is not None and output_channels == input_channels:
        if residual_learning == 'add':
            output = tf.keras.layers.add([img_input, layer])
        else:
            output = tf.keras.layers.subtract([img_input, layer])
    else:
        output = layer

    name = 'DnCNN_' + str(id) + '_d' + str(depth)
    return tf.keras.Model(inputs=img_input, outputs=output, name=name)


def ffdnet(mode='grayscale', residual_learning=True, scale=2, num_filters=None, depth=None, kernel_size=3,
           weight_decay=0, maps_channels=1, Norm='Bnorm', NormParams=[], SelfSimBlocks=None):
    if mode == 'grayscale':
        num_filters = 64 if num_filters is None else num_filters
        depth = 13 if depth is None else depth
        image_channels = 1
    elif mode == 'RGB':
        num_filters = 96 if num_filters is None else num_filters
        depth = 10 if depth is None else depth
        image_channels = 3
    else:
        raise Exception('Invalid mode for FFDNet input images. Got: ' + str(mode))

    img_input = tf.keras.Input(shape=[None, None, image_channels])
    map_input = tf.keras.Input(shape=[None, None, maps_channels])

    layer = PixelShuffle(scale=scale)([img_input, map_input])

    dncnn_model = dncnn(mode=mode, residual_learning=False,
                        num_filters=num_filters, depth=depth,
                        input_channels=image_channels * scale * scale + maps_channels,
                        output_channels=image_channels * scale * scale,
                        kernel_size=kernel_size,
                        weight_decay=weight_decay,
                        Norm=Norm,
                        NormParams=NormParams,
                        SelfSimBlocks=SelfSimBlocks)
    layer = dncnn_model(layer)

    layer = PixelUnShuffle(scale=scale)(layer)

    if residual_learning is not None:
        if residual_learning == 'add':
            output = tf.keras.layers.add([img_input, layer])
        else:
            output = tf.keras.layers.subtract([img_input, layer])
    else:
        output = layer

    return tf.keras.Model(inputs=[img_input, map_input], outputs=output, name='FFDNet')


def load_weights_mat(mat_path, model, verbose=False):
    tot_par = 0

    try:
        network = sio.loadmat(mat_path)
        layers = network['net'][0, 0][0][0, :]

        i = 0
        for layer in layers:
            if layer['type'] == 'conv':
                while not (isinstance(model.layers[i], tf.keras.layers.Conv2D)):
                    i += 1

                kernel = layer['weights'][0][0][0][0]
                bias = layer['weights'][0][0][0][1][:, 0]
                if kernel.ndim == 3:
                    kernel = np.expand_dims(kernel, -1)

                model.layers[i].set_weights([kernel, bias])

                tot_par = tot_par + tf.reduce_prod(kernel.shape)
                tot_par = tot_par + tf.reduce_prod(bias.shape)

                if verbose:
                    print('{0} -> {1}'.format(layer['name'], model.layers[i].name))

                i += 1
    except:
        with h5py.File(mat_path, 'r') as hf:
            i = 0
            for layer in hf['/net/layers']:

                ltype = ''.join([chr(x) for x in hf[layer[0]]['type'][:][:, 0]])  # Type decoding

                if ltype == 'conv':
                    while not (isinstance(model.layers[i], tf.keras.layers.Conv2D)):
                        i += 1

                    kernel = np.transpose(np.array(hf[hf[layer[0]]['weights'][0, 0]]))
                    bias = np.transpose(np.array(hf[hf[layer[0]]['weights'][1, 0]]))[:, 0]
                    if kernel.ndim == 3:
                        kernel = np.expand_dims(kernel, -1)

                    model.layers[i].set_weights([kernel, bias])

                    tot_par = tot_par + tf.reduce_prod(kernel.shape)
                    tot_par = tot_par + tf.reduce_prod(bias.shape)

                    if verbose:
                        lname = ''.join([chr(x) for x in hf[layer[0]]['name'][:][:, 0]])  # Name decoding
                        print('{0} -> {1}'.format(lname, model.layers[i].name))

                    i += 1

    print('Parameters loaded: {0}'.format(tot_par))
