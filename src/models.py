import tensorflow as tf

from self_sim_block import get_self_sim_block_residual
from ffdnet import dncnn, ffdnet


def get_model(base_model='dncnn', mode='grayscale', residual_learning=True, sub_residual_learning=True, num_filters=64,
              depth=13, kernel_size=3, weight_decay=0, Norm=None, NormParams=[], scale=2, SelfSimBlocks=None,
              verbose=False):
    if mode == 'grayscale':
        image_channels = 1
    elif mode == 'RGB':
        image_channels = 3
    else:
        raise Exception('Invalid mode for input images. Got: ' + str(mode))

    if (SelfSimBlocks is None) or (SelfSimBlocks is not None and SelfSimBlocks['block_model'] == 'inner'):

        if base_model == 'dncnn':
            model = dncnn(mode=mode,
                          residual_learning=residual_learning,
                          num_filters=num_filters,
                          depth=depth,
                          kernel_size=kernel_size,
                          weight_decay=weight_decay,
                          Norm=Norm,
                          NormParams=NormParams,
                          SelfSimBlocks=SelfSimBlocks)

        elif base_model == 'ffdnet':
            model = ffdnet(mode=mode,
                           residual_learning=residual_learning,
                           scale=scale,
                           num_filters=num_filters,
                           depth=depth,
                           kernel_size=kernel_size,
                           weight_decay=weight_decay,
                           Norm=Norm,
                           NormParams=NormParams,
                           SelfSimBlocks=SelfSimBlocks)
        else:
            raise Exception('Invalid base_model')

    elif SelfSimBlocks['block_model'] == 'outer':  # Blocks are outside the dncnn/ffdnet sub-models

        img_input = tf.keras.Input(shape=[None, None, image_channels])
        layer = img_input

        pos = SelfSimBlocks['block_pos']
        if pos[-1] >= depth:
            raise Exception('Last block position greater than network depth')

        prev_pos = 0
        for block_id in range(len(pos)):
            sub_depth = pos[block_id] - prev_pos

            print('block_id: {}'.format(block_id))
            print('sub_depth: {}'.format(sub_depth))

            if sub_depth < 0:
                raise Exception('Block positions not sorted')
            else:
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

                if sub_depth > 0:
                    input_channels = num_filters if block_id > 0 else None
                    output_channels = num_filters

                    if base_model == 'dncnn':
                        sub_model = dncnn(mode=mode,
                                          residual_learning=sub_residual_learning,
                                          num_filters=num_filters,
                                          depth=sub_depth,
                                          input_channels=input_channels,
                                          output_channels=output_channels,
                                          kernel_size=kernel_size,
                                          weight_decay=weight_decay,
                                          Norm=Norm,
                                          NormParams=NormParams,
                                          SelfSimBlocks=None,
                                          id=block_id)
                    else:
                        raise Exception('Invalid base_model')

                    layer = sub_model(layer)

                layer = get_self_sim_block_residual(layer,
                                                    id='SelfSimBlock' + str(block_id),
                                                    hnsz=next_hnsz,
                                                    stride=next_stride,
                                                    Norm=SelfSimBlocks['Norm'],
                                                    NormParams=SelfSimBlocks['NormParams'],
                                                    weight_decay=SelfSimBlocks["weight_decay"],
                                                    patch_folding_size=SelfSimBlocks["patch_folding_size"],
                                                    shift_pad=SelfSimBlocks['shift_pad'],
                                                    verbose=verbose)(layer)

            prev_pos = pos[block_id]

        res_depth = depth - prev_pos
        print('res_depth: {}'.format(res_depth))
        if res_depth <= 0:
            raise Exception('The network must ends with convolutions')

        if base_model == 'dncnn':
            sub_model = dncnn(mode=mode,
                              residual_learning=sub_residual_learning,
                              num_filters=num_filters,
                              depth=res_depth,
                              input_channels=num_filters,
                              output_channels=image_channels,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              Norm=Norm,
                              NormParams=NormParams,
                              SelfSimBlocks=None,
                              id=block_id + 1)
        else:
            raise Exception('Invalid base_model')

        layer = sub_model(layer)

        if residual_learning is not None:
            if residual_learning == 'add':
                output = tf.keras.layers.add([img_input, layer])
            else:
                output = tf.keras.layers.subtract([img_input, layer])
        else:
            output = layer

        model = tf.keras.Model(inputs=img_input, outputs=output)

    return model
