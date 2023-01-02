# Self-Similarity Network (SSN) for Image Denoising
The repository is organized as follows:

* `src` contains the training and testing code.
* `weights` contains the weights of some trained models for specific noise levels.

### Training set preparation
First of all we need to prepare the training set, using the `datasets.py` script. For instance, if `datasets/BSD500` contains the 400 images from the train and test splits of BSD500, to produce the dataset used for training the released models, we can run

```
python datasets.py config_dataset.json datasets/BSD500 -o bsd500grayscale
```
which produces a `bsd500grayscale.tfrec` file containing the training patches.
### Training

Provided the training set `bsd500grayscale.tfrec` is stored in a `datasets` directory, a training can be launched with the following command:
```
python train.py config_train.json datasets/bsd500grayscale.tfrec
```

### Evaluation and testing
The configuration file `config_eval.json` specifies the architecture of the model to be evaluated and the noise level. The following command evaluates a model on the three test sets using the pre-trained weights stored in `weights/sigma25`.

```
python evaluate.py config_eval.json weights/sigma25 datasets/Set12 datasets/BSD68 datasets/Urban100
```
