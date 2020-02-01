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
import matplotlib.pyplot as plt
import pandas as pd

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
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x), axis=axis, keepdims=True)


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

    def backward(self, delta, reg, gamma, lr, do_gd = True):
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
        self.w = np.random.normal(loc=0, scale=1/np.sqrt(in_units), size=(in_units, out_units))    # Declare the Weight matrix
        self.b = np.zeros(out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.v_dw = np.zeros_like(self.d_w) #xp
        self.v_db = np.zeros_like(self.d_b) #xp

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

    def backward(self, delta,reg, gamma,lr, do_gd = True):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = delta.dot(self.w.T) 
        self.d_w = self.x.T.dot(delta) / delta.shape[0] + reg*self.w/ delta.shape[0]
        self.d_b = delta.sum(axis=0) / delta.shape[0] + reg*self.b/ delta.shape[0]
        if do_gd:
            self.v_dw = gamma*self.v_dw + lr*self.d_w
            self.v_db = gamma*self.v_db + lr*self.d_b
            self.w += self.v_dw #xp
            self.b += self.v_db #xp
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
        self.reg = config['L2_penalty']
        self.gamma = config['momentum_gamma']
        self.lr = config['learning_rate']
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
        reg_loss = 0
        for layer in self.layers:
            if type(layer) is Layer:
                reg_loss += np.sum(layer.w*layer.w) + np.sum(layer.b*layer.b)
        
        # clip is used to avoid log(0)
        loss = -(targets*np.log(np.clip(logits,a_min=1e-10, a_max=None))).sum()/targets.shape[0] + self.reg*reg_loss/(2*targets.shape[0])
        return loss

    def backward(self,do_gd = True):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.reg, self.gamma, self.lr, do_gd)
        return delta


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    batch_size = config['batch_size']
    EPOCHS = config['epochs']
    batches = x_train.shape[0]//batch_size
    train_errors = []
    valid_errors = []
    train_accuracies = []
    valid_accuracies = []
    layers = len(model.layers)//2+1
    best_w = [0 for _ in range(layers)]
    best_b = [0 for _ in range(layers)]

    early_stop_counter = 0
    for epoch in range(EPOCHS):
        shuffle_index = list(np.random.permutation(x_train.shape[0]))
        x_train = x_train[shuffle_index,:]
        y_train = y_train[shuffle_index]
        for i in range(batches):
            input_train = x_train[batch_size*i:batch_size*(i+1),:]
            target_train = y_train[batch_size*i:batch_size*(i+1),:]
            model.forward(input_train,targets=target_train)
            model.backward()
        (train_out,train_error) = model.forward(x_train, targets=y_train)
        (valid_out,valid_error) = model.forward(x_valid, targets=y_valid)
        if len(valid_errors)!=0 and valid_error > valid_errors[-1]:
            early_stop_counter += 1
        else:
            early_stop_counter = 0

        train_accuracies.append(np.sum(np.argmax(train_out,axis = 1)==np.argmax(y_train,axis = 1))/(y_train.shape[0]))
        valid_accuracies.append(np.sum(np.argmax(valid_out,axis = 1)==np.argmax(y_valid,axis = 1))/(y_valid.shape[0]))
        train_errors.append(train_error)
        valid_errors.append(valid_error)
        if config['early_stop'] and early_stop_counter == config['early_stop_epoch']:
            print("early stopping at epoch {}, best performance at epoch {}.".format(epoch, epoch - early_stop_counter))
            break
        else:
            for j in range(layers):
                best_w[j] = model.layers[2*j].w 
                best_b[j] = model.layers[2*j].b

    for j in range(layers):
        model.layers[2*j].w = best_w[j]
        model.layers[2*j].b = best_b[j]
    return train_errors, valid_errors, train_accuracies, valid_accuracies


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    (output,_) = model.forward(X_test)
    pred = np.argmax(output,axis = 1)
    label = np.argmax(y_test,axis = 1)
    acc = np.sum(pred== label)/(y_test.shape[0])
    return acc


def check_grad(model, X_check, y_check):
    X_check =X_check[:1]
    y_check =y_check[:1]
    table = []
    epsilon = 1e-2
    # output bias check
    # hidden bias check
    # hidden to output check
    # input to hidden check
    check_list = ['model.layers[-1].d_b[0]', 'model.layers[-3].d_b[0]', 'model.layers[-1].d_w[0][0]', 'model.layers[-1].d_w[1][1]',\
        'model.layers[2].d_w[0][0]', 'model.layers[2].d_w[1][1]']
    
    for grad in check_list:
        model.forward(X_check, targets=y_check)
        model.backward(do_gd = False)
        target = grad.replace('d_','')
        x=[target, epsilon, -eval(grad), None, None]
        exec(target+' -= epsilon')
        pre = model.forward(X_check, targets=y_check)[1]
        exec(target+' += 2*epsilon')
        curr = model.forward(X_check, targets=y_check)[1]
        x[3] = (curr - pre)/epsilon/2
        x[4] = abs(x[3] - x[2])
        table.append(x)
    
    df = pd.DataFrame(table, columns=['target','epsilon','gradient','approx', 'delta'])
    print(df)

def split(X, y):
    # Create splits for validation data here.
    train_indices = []
    valid_indices = []

    # evenly distribute labels
    for i in range(0, y.shape[1]):
        total = np.argwhere(y[:,i]==1).flatten()
        valid_indices.extend(total[:1000])
        train_indices.extend(total[1000:])

    assert sum(valid_indices)+sum(train_indices) == sum(range(len(X)))
    x_valid, y_valid = X[valid_indices], y[valid_indices]
    x_train, y_train = X[train_indices], y[train_indices]
    return x_train, y_train, x_valid, y_valid

def getCheck(X, y):
    check_indices = []
    for i in range(0, y.shape[1]):
        total = np.argwhere(y[:,i]==1).flatten()
        check_indices.extend(total[:1])
    return X[check_indices], y[check_indices]


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    x_train, y_train, x_valid, y_valid = split(x_train, y_train)
    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)
    
    test_acc = test(model, x_test, y_test)
    print("Test accuracy:" + str(test_acc))
