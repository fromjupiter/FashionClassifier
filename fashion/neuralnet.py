################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    #min-max norm
    imin = img.min(axis=1)
    imax = img.max(axis=1)
    img = (img.T - imin)/(imax-imin)
    return img.T


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    res = np.zeros((len(labels), num_classes))
    for i, l in enumerate(labels):
        res[i][l] = 1
    
    # recover by np.argmax(res, axis=1)
    return res


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x, axis=1):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    # avoid overflow
    x -= np.max(x, axis=axis)
    return np.exp(x)/np.sum(np.exp(x), axis=axis)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        self.x = a
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return x * (x > 0)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x)*(1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1.0/np.square(np.cosh(self.x))

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return self.x>0


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.normal(size=(in_units, out_units))    # Declare the Weight matrix
        self.b = np.zeros(out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x.dot(self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = delta.dot(self.w.T)
        self.d_w = self.x.T.dot(delta)
        self.d_b = delta
        return self.d_x


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        if targets is not None:
            self.targets = targets
        self.x = x
        for layer in self.layers:
            x = layer.forward(x)
        self.y = softmax(x, axis=1)
        return (self.y, self.loss(self.y, targets) if targets is not None else None)

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        return -np.log(logits).dot(targets)

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    train_indices = []
    valid_indices = []

    # evenly distribute labels
    for i in range(0, y_train.shape[1]):
        total = np.argwhere(y_train[:,i]==1).flatten()
        valid_indices.extend(total[:1000])
        train_indices.extend(total[1000:])

    assert sum(valid_indices)+sum(train_indices) == sum(range(len(x_train)))
    x_valid, y_valid = x_train[valid_indices], y_train[valid_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
