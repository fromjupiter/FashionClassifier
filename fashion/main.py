import argparse
import matplotlib.ticker as mticker
from neuralnet import *


def trainModelByConfig(config, plot_title='Default Title'):
    model  = Neuralnetwork(config)
    # Load the data
    X, y = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")
    print('--- Below is the routine output ---')
    X_train, y_train, X_valid, y_valid = split(X, y, do_check=False)
    
    train_errors,valid_errors, train_accuracies, valid_accuracies = train(model, X_train, y_train, X_valid, y_valid, config)
    print("Test accuracy:" + str(test(model, x_test, y_test)))
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle(plot_title)
    axs[0].set_title('Cross-Entropy Loss')
    axs[1].set_title('Accuracy(Percent Correct)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(range(len(train_errors)))
    xlabels = list(range(len(train_errors)))
    axs[0].plot(xlabels, train_errors,color = 'b', label='train')
    axs[0].plot(xlabels, valid_errors,color = 'g', label='holdout')
    axs[0].legend()
    axs[1].plot(xlabels, train_accuracies,color = 'b', label='train')
    axs[1].plot(xlabels, valid_accuracies,color = 'g', label='holdout')
    axs[1].legend()
    plt.draw()
    plt.pause(0.001)

def showTraining():
    config = load_config("./")
    trainModelByConfig(config, plot_title='NN Training')
    # avoid quitting
    plt.show()

def showRegularization():
    config = load_config("./")
    config['L2_penalty'] = 0.001
    trainModelByConfig(config, plot_title='L2=0.001')
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
    pass

parser = argparse.ArgumentParser(description='Fashion Classifier entry point')

parser.add_argument('-r', '--routine', default='showTraining', help='Run predefined routine')

args = parser.parse_args()


# Create the model

locals()[args.routine]()