## ml_defense: Attacks and Defenses for Machine Learning Systems

### Overview
This library is written purely in Python 2. It can be used to reproduce and evaluate white-box attacks on machine learning systems (support vector machines and neural networks) by generating adversarial examples for different datasets. But more importantly, it demonstrates multiple defenses using dimensionality reduction or equivalent techniques against the mentioned attacks as well as defense-aware attacks. The detailed explanation of the defenses can be found in this [paper](https://128.84.21.199/abs/1704.02654).

Please note that the github repository does not contain any dataset or trained model. When the code is run for the first time, the required dataset will be downloaded and the specified model will be trained and saved automatically.

### System Requirements
The use of this library for attacks and defenses for support vector machines (SVMs) requires the installation of [numpy](http://www.numpy.org/) and [scikit-learn](http://scikit-learn.org/stable/). For experiments with neural networks, [Theano](http://deeplearning.net/software/theano/) and [lasagne](https://github.com/Lasagne/Lasagne) need to be installed as well. Theano can be configured to work with a GPU.

The code has been tested with the following versions of dependencies:
- numpy:        1.12.1
- scikit-learn: 0.18.1
- Theano:       0.9.0
- Lasagne:      0.2.dev1

### Basic Usage
#### Neural Networks
To demonstrate an attack on an original classifier and a classifier implemented with one of the defense mechanisms, run `run_defense.py`:
```
python run_defense.py [-m MODEL] [--dataset DATASET] [-c N_CHANNELS] [--n_epoch N_EPOCHS]
                      [-a ATTACK] [-d DEFENSE] [-dr DIMENSION_REDUCTION]
```
All the arguments have a default value. For more detail on the arguments, type:
```
python run_defense.py --help
```
For a trial of the effect of the reconstruction defense using the Principal Component Analysis (PCA) as dimensionality reduction against the Fast Gradient Attack on a 2-layer multilayer perceptron trained on the [MNIST handwritten digit](http://yann.lecun.com/exdb/mnist/) dataset,  simply run:
```
python run_defense.py
```
For a demonstration of a defense-aware attack (strategic attack), you can run the command with the same set of arguments but change the script from `run_defense.py` to  `strategic_attack_demo.py`:
```
python strategic_attack_demo.py [-m MODEL] [--dataset DATASET] [-c N_CHANNELS]
                                [--n_epoch N_EPOCHS] [-a ATTACK] [-d DEFENSE]
                                [-dr DIMENSION_REDUCTION]
```

A trained neural network with 2 hidden layers of width 100, and sigmoid activations has been included in the [nn_models](./nn_models) folder.

The script `strategic_attack_demo.py` (as well as the other scripts) will provide as output the performance of the ML system on adversarial inputs, with and without defenses. Both accuracy and confidence values will be stored in a .txt file in the output folder. In case of image datasets, for each perturbation value used, the first ten images will be stored in the visual data folder.

#### SVM
To demonstrate a strategic attack on SVM with retrain defense, run the following command:
```
python strategic_svm.py [--dataset DATASET] [-c N_CHANNELS] [--two_classes] [-C PENALTY_CONST]
                        [-p PENALTY_NORM] [-dr DIMENSION_REDUCTION]
```
If `--two_classes` flag is included, SVM will be trained only on two classes of the training set instead of multi-class.

The [lib](./lib) folder contains the implementation of the attacks and defenses, as well as the various utilities required.

### Datasets
This library currently supports the use of the following datasets:  
- `[--dataset MNIST]` MNIST (default)
- `[--dataset GTSRB]` German Traffic Sign Recognition Benchmark

### Attacks
Attacks algorithm supported at this time are:
- `[-a fg]` Fast Gradient method (default)
- `[-a fsg]` Fast Sign Gradient method

### Defenses
The detail on the defense can be found in the paper. Supported defense mechanisms are:
- `[-d recons]` Reconstruction defense (default)
- `[-d retrain]` Retrain defense

### Plotting

### Contributors
Arjan Bhagoji (abhagoji@princeton.edu)  
Chawin Sitawarin (chawins@princeton.edu)  
Electrical Engineering Department, Princeton University

This repository is under active development and we welcome contributions.
