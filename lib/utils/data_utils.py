import sys,os
import numpy as np

#------------------------------------------------------------------------------#
#Function to load MNIST data
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename,abs_path, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        print filename
        urlretrieve(source + filename, abs_path+filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(abs_path,filename):
        if not os.path.exists(abs_path+filename):
            download(filename,abs_path)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(abs_path+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(abs_path,filename):
        if not os.path.exists(abs_path+filename):
            download(filename,abs_path)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(abs_path+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                                                    os.path.abspath(__file__))))
    rel_path="input_data/"
    abs_path=os.path.join(script_dir,rel_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    X_train = load_mnist_images(abs_path,'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(abs_path,'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(abs_path,'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(abs_path,'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test
#------------------------------------------------------------------------------#
