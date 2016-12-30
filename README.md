## ml_defense: Attacks and defenses for machine learning systems

### System requirements
This library is written in Python. The use of this library for attacks and defenses for support vector machines (SVMs) requires the installation of [numpy](http://www.numpy.org/) and [scikit-learn](http://scikit-learn.org/stable/). For experiments with neural networks, [Theano](http://deeplearning.net/software/theano/) and [lasagne](https://github.com/Lasagne/Lasagne) need to be installed as well. Theano can be configured to work with a GPU.

### Basic usage
For a simple trial of the reconstruction defense on the [MNIST handwritten digit](http://yann.lecun.com/exdb/mnist/) dataset, run `recons_nns.py` with the model name set to 'mlp' and the model flag set to 1:
```
python recons_nns.py -m mlp -f 1
```

A trained neural network with 2 hidden layers of width 100, and sigmoid activations has been included in the [models](./models) folder. To train a custom neural network, change the  string after the `-m` option and set the `-f` option to 0.

The script `recons_nns.py` will provide as output the performance of the ML system on adversarial inputs, with and without defenses. Both accuracy and confidence values will be stored in a .txt file in the output folder.

The [lib](./lib) folder contains the implementation of the attacks and defenses.

### Datasets

### ML systems

### Plotting
