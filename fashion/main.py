import argparse
import matplotlib.ticker as mticker
from neuralnet import *


def trainModelByConfig(config, plot_title='Default Title'):
    model  = Neuralnetwork(config)
    # Load the data
    X, y = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")
    X_train, y_train, X_valid, y_valid = split(X, y)
    
    train_errors,valid_errors, train_accuracies, valid_accuracies = train(model, X_train, y_train, X_valid, y_valid, config)
    print("Test accuracy:" + str(test(model, x_test, y_test)))
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle(plot_title)
    axs[0].set_title('Cross-Entropy Loss')
    axs[1].set_title('Accuracy(Percent Correct)')
    axs[1].set_xlabel('Epoch')
    xlabels = list(range(len(train_errors)))
    axs[0].plot(xlabels, train_errors,color = 'b', label='train')
    axs[0].plot(xlabels, valid_errors,color = 'g', label='holdout')
    axs[0].legend()
    axs[1].plot(xlabels, train_accuracies,color = 'b', label='train')
    axs[1].plot(xlabels, valid_accuracies,color = 'g', label='holdout')
    axs[1].legend()
    plt.draw()
    plt.pause(0.001)

def checkGrad():
    config = load_config("./")
    model  = Neuralnetwork(config)
    X, y = load_data(path="./", mode="train")
    X_check, y_check = getCheck(X, y)
    print("---- Checking gradient ----")
    check_grad(model, X_check, y_check)
    print("---- Checking done ----")

def showTraining():
    config = load_config("./")
    trainModelByConfig(config, plot_title='NN Training')
    # avoid quitting
    plt.show()

def showRegularization():
    config = load_config("./")
    config['epochs'] = int(config['epochs'] * 1.1)
    print("Using 0.001 L2 regularization..")
    config['L2_penalty'] = 0.001
    trainModelByConfig(config, plot_title='L2=0.001')
    print("Using 0.0001 L2 regularization..")
    config['L2_penalty'] = 0.0001
    trainModelByConfig(config, plot_title='L2=0.0001')
    # avoid quitting
    plt.show()

def showActivation():
    config = load_config("./")
    print("Using ReLU as activation..")
    config['activation'] = 'ReLU'
    trainModelByConfig(config, plot_title='Activation=ReLU')
    print("Using tanh as activation..")
    config['activation'] = 'tanh'
    trainModelByConfig(config, plot_title='Activation=Tanh')
    print("Using sigmoid as activation..")
    config['activation'] = 'sigmoid'
    trainModelByConfig(config, plot_title='Activation=Sigmoid')
    # avoid quitting
    plt.show()

def showNetworkTopology():
    config = load_config("./")
    print("Halving hidden units..")
    config['layer_specs'][1] = 25
    trainModelByConfig(config, plot_title='Half Hidden Units(25)')
    print("Doubling hidden units..")
    config['layer_specs'][1] = 100
    trainModelByConfig(config, plot_title='Double Hidden Units(100)')
    print("Using two hidden layers..")
    config['layer_specs'] = [784, 47, 47, 10]
    trainModelByConfig(config, plot_title='Two Hidden Layers')
    plt.show()

parser = argparse.ArgumentParser(description='Fashion Classifier entry point')

parser.add_argument('-r', '--routine', default='showTraining', help='Run predefined routine')

args = parser.parse_args()


# Create the model

print('--- Below is the routine output ---')
locals()[args.routine]()