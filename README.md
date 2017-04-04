## ml_defense: Attacks and defenses for machine learning systems

### System requirements
This library is written in Python. The use of this library for attacks and defenses for support vector machines (SVMs) requires the installation of [numpy](http://www.numpy.org/) and [scikit-learn](http://scikit-learn.org/stable/). For experiments with neural networks, [Theano](http://deeplearning.net/software/theano/) and [lasagne](https://github.com/Lasagne/Lasagne) need to be installed as well. Theano can be configured to work with a GPU.

### Basic usage
For a trial of the effect of the defense on the [MNIST handwritten digit](http://yann.lecun.com/exdb/mnist/) dataset against the Fast Gradient Sign Attack, run `strategic_attack_demo.py`:
```
python strategic_attack_demo.py
```

A trained neural network with 2 hidden layers of width 100, and sigmoid activations has been included in the [nn_models](./nn_models) folder.

The script `strategic_attack_demo.py` will provide as output the performance of the ML system on adversarial inputs, with and without defenses. Both accuracy and confidence values will be stored in a .txt file in the output folder. In case of image datasets, for each perturbation value used, the first ten images will be stored in the visual data folder.

The [lib](./lib) folder contains the implementation of the attacks and defenses, as well as the various utilities required.

### Datasets
The codebase currently supports the use of the [MNIST] and 

### ML systems

### Plotting
