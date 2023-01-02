import argparse
import glob
import json
import pickle

import matplotlib.pyplot as plt
from PIL import Image

from datasets import read_img, pad_image, unpad_image
from models import get_model
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Evaluation of SelfSimNet")

    parser.add_argument("config", help="Configuration json file")
    parser.add_argument("weights", help="Weights of a model compatible with hyperparameters in config file")
    parser.add_argument("testing_sets", nargs='*', help="Directory with testing images")
    parser.add_argument("-v", "--verbosity", type=int)
    parser.add_argument("-s", "--sigma", type=int)
    parser.add_argument("-i", "--image")

    args = parser.parse_args()
    verbosity = args.verbosity
    if verbosity is None:
        verbosity = 0
    weights = args.weights
    sets = args.testing_sets

    print(args)

    with open(args.config, "r") as cf:
        config = json.load(cf)

    print(config)

    base_model = config["base_model"]
    scale = config["scale"]
    mode = config["mode"]
    sigmas = config["sigmas"]
    residual_learning = config["residual_learning"]
    sub_residual_learning = config["sub_residual_learning"]
    num_filters = config["num_filters"]
    kernel_size = config["kernel_size"]
    block = config["block"]
    depth = config["depth"]
    Norm = config["Norm"]
    NormParams = config["NormParams"]

    shift_pad = config["SelfSimBlocks"]["shift_pad"]
    block_pos = config["SelfSimBlocks"]["block_pos"]
    block_model = config["SelfSimBlocks"]["block_model"]
    hnsz = config["SelfSimBlocks"]["hnsz"]
    stride = config["SelfSimBlocks"]["stride"]
    block_norm = config["SelfSimBlocks"]["Norm"]
    block_normparams = config["SelfSimBlocks"]["NormParams"]

    config["SelfSimBlocks"]["weight_decay"] = -1

    if block is not None:
        SelfSimBlocks = config["SelfSimBlocks"]
    else:
        SelfSimBlocks = None

    if verbosity >= 1:
        print("\nMODEL HYPERPARAMETERS")
        print("base_model: {}".format(base_model))
        print("scale: {}".format(scale))
        print("mode: {}".format(mode))
        print("residual_learning: {}".format(residual_learning))
        print("sub_residual_learning: {}".format(sub_residual_learning))
        print("num_filters: {}".format(num_filters))
        print("kernel_size: {}".format(kernel_size))
        print("block: {}".format(block))
        print("shift_pad: {}".format(shift_pad))
        print("depth: {}".format(depth))
        print("block_pos: {}".format(block_pos))
        print("block_model: {}".format(block_model))
        print("hnsz: {}".format(hnsz))
        print("stride: {}".format(stride))
        print("Norm: {}".format(Norm))
        print("NormParams: {}".format(NormParams))
        print("sigmas: {}".format(sigmas))

        print("block_norm: {}".format(block_norm))
        print("block_normparams: {}".format(block_normparams))

    for ts in sets:
        if not os.path.isdir(ts):
            print('{} not found'.format(ts))
            exit(0)

    print('\nWeights: {}'.format(weights))
    print("Testing sets: {}".format(sets))

    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))

    get_confirmation('Continue?')


    # PREPARE MODEL

    model = get_model(base_model=base_model,
                      mode=mode,
                      residual_learning=residual_learning,
                      sub_residual_learning=sub_residual_learning,
                      num_filters=num_filters,
                      depth=depth,
                      kernel_size=kernel_size,
                      Norm=Norm,
                      NormParams=NormParams,
                      scale=scale,
                      SelfSimBlocks=SelfSimBlocks,
                      verbose=verbosity)

    # Load weights
    if weights.split('.')[-1] == 'mat':
        from ffdnet import load_weights_mat
        if base_model == 'ffdnet':
            load_weights_mat(weights, model.layers[3], verbosity)
        else:
            load_weights_mat(weights, model, verbosity)
    else:
        model.load_weights(weights)

    # Set or unset noise map  flag
    if base_model == 'ffdnet':
        noise_map_input = True
    else:
        noise_map_input = False

    # What to do?
    if args.image:
        tf.random.set_seed(1234)
        print('Applying model on image: {}'.format(args.image))
        clean_img, noisy_img, img_hat = denoise_random(model, img_path=args.image, mode=mode, sigma=args.sigma,
                                                       noise_map_input=noise_map_input, scale=scale, verbose=verbosity)

        # clip images
        img_hat = tf.clip_by_value(img_hat, 0, 1)
        noisy_img = tf.clip_by_value(noisy_img, 0, 1)

        print('Noisy\nPSNR: {0}\nSSIM: {1}'.format(tf.image.psnr(clean_img, noisy_img, 1).numpy(),
                                                   tf.image.ssim(clean_img, noisy_img, 1).numpy()))
        print('Denoised\nPSNR: {0}\nSSIM: {1}'.format(tf.image.psnr(clean_img, img_hat, 1).numpy(),
                                                      tf.image.ssim(clean_img, img_hat, 1).numpy()))

        noisy_img = Image.fromarray(noisy_img.numpy()[:, :, 0] * 255)
        noisy_img = noisy_img.convert('L')
        noisy_img.save('noisy.png')
        img_hat = Image.fromarray(img_hat.numpy()[:, :, 0] * 255)
        img_hat = img_hat.convert('L')
        img_hat.save('denoised.png')

        return

    if args.sigma:
        print('Producing sample images with sigma={}'.format(args.sigma))

        files = []

        for ds in sets:
            files += glob.glob(os.path.join(ds, '*'))

        print(files)

        visualize(model, mode=mode, files=files, sigma=args.sigma, noise_map_input=noise_map_input, scale=scale,
                  verbose=verbosity)
        return


    # Evaluate
    print('Evaluating model on testsets')

    results = evaluate(model, mode=mode, testsets=sets, sigmas=sigmas, noise_map_input=noise_map_input, scale=scale,
                       verbose=verbosity)

    summarize_results(results)

    # Save results
    model_name = weights.split('/')[-1].split('.')[0]
    with open('eval_' + model_name, 'wb') as f:
        pickle.dump(results, f)


def denoise_random(model, img_path, mode, sigma, noise_map_input=False, scale=1, verbose=0):
    clean_img = read_img(img_path, mode)
    clean_img = tf.constant(clean_img)

    # padding
    padded_clean_img = pad_image(clean_img, scale)

    noise = tf.random.normal(padded_clean_img.shape, 0, sigma)
    noise = noise / 255.0
    noisy_img = tf.add(padded_clean_img, noise)

    noisy_img = tf.expand_dims(noisy_img, axis=0)

    # prediction
    if noise_map_input:
        noise_map = tf.fill((1, noisy_img.shape[1], noisy_img.shape[2], 1), sigma / 255.0)

        img_hat = model.predict([noisy_img, noise_map], verbose=verbose)[0, :, :, :]
    else:
        img_hat = model.predict(noisy_img, verbose=verbose)[0, :, :, :]

    # unpad
    img_hat = unpad_image(img_hat, clean_img.shape[0], clean_img.shape[1])
    noisy_img = unpad_image(noisy_img[0, :, :, :], clean_img.shape[0], clean_img.shape[1])

    return clean_img, noisy_img, img_hat


def evaluate(model, mode, testsets, sigmas, noise_map_input=False, scale=1, verbose=0):

    results = {}
    for testset in testsets:

        sigma_dic = {}
        for sigma in sigmas:
            tf.random.set_seed(1234)

            if verbose:
                print('Testset: {0}. Sigma: {1}'.format(testset, sigma))

            files = glob.glob(os.path.join(testset, '*'))

            files_dic = {}
            for f in files:
                clean_img, _, img_hat = denoise_random(model, img_path=f, mode=mode, sigma=sigma,
                                                       noise_map_input=noise_map_input, scale=scale, verbose=verbose)

                # compute PSNR and SSIM
                psnr = tf.image.psnr(clean_img, tf.clip_by_value(img_hat, 0, 1), 1)
                ssim = tf.image.ssim(clean_img, tf.clip_by_value(img_hat, 0, 1), 1)

                files_dic[f.split('/')[-1]] = {'psnr': psnr.numpy(), 'ssim': ssim.numpy()}

            sigma_dic[sigma] = files_dic

        results[testset.split('/')[-1]] = sigma_dic

    return results


def summarize_results(results):
    for set in results.keys():
        print(set)
        for sigma in results[set].keys():
            n = len(results[set][sigma].keys())
            avg_psnr = 0
            avg_ssim = 0
            for img in results[set][sigma].keys():
                avg_psnr += results[set][sigma][img]['psnr']
                avg_ssim += results[set][sigma][img]['ssim']

            avg_psnr /= n
            avg_ssim /= n

            print('sigma: {0}; average PSNR: {1}, average SSIM: {2}'.format(sigma, avg_psnr, avg_ssim))


def visualize(model, mode, files, sigma, noise_map_input=False, scale=1, verbose=0):
    tf.random.set_seed(1234)

    for f in files:
        clean_img, noisy_img, img_hat = denoise_random(model, img_path=f, mode=mode, sigma=sigma,
                                                       noise_map_input=noise_map_input, scale=scale, verbose=verbose)

        # clip images
        img_hat = tf.clip_by_value(img_hat, 0, 1)
        noisy_img = tf.clip_by_value(noisy_img, 0, 1)

        plt.figure(figsize=(8.4, 3.5))
        plt.subplot(1, 3, 1)
        plt.imshow(clean_img[:, :, 0], cmap='gray')
        plt.title('Clean', color='black')
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_img[:, :, 0], cmap='gray')
        plt.title('Noisy\nPSNR: {0:.4f}\nSSIM: {1:.4f}'.format(tf.image.psnr(clean_img, noisy_img, 1).numpy(),
                                                               tf.image.ssim(clean_img, noisy_img, 1).numpy()),
                  color='black')
        plt.subplot(1, 3, 3)
        plt.imshow(img_hat[:, :, 0], cmap='gray')
        plt.title('Denoised\nPSNR: {0:.4f}\nSSIM: {1:.4f}'.format(tf.image.psnr(clean_img, img_hat, 1).numpy(),
                                                                  tf.image.ssim(clean_img, img_hat, 1).numpy()),
                  color='black')
        print(tf.image.psnr(clean_img, img_hat, 1).numpy())
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        os.makedirs('samples', exist_ok=True)
        plt.tight_layout()
        plt.savefig('samples/' + f.split('/')[-1].split('.')[0] + '.pdf')


if __name__ == '__main__':
    main()
