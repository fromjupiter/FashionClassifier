Neural Networks on Fashion MNIST datatset


## Get started
Before running the script, you should make sure that config.yaml and data (compressed gz files) are under current directory. 
> ./ <br/>
> ├─README.md <br/>
> ├─config.yaml <br/>
> ├─t10k-images-idx3-ubyte.gz <br/>
> ├─t10k-labels-idx1-ubyte.gz <br/>
> ├─train-images-idx3-ubyte.gz <br/>
> ├─train-labels-idx1-ubyte.gz <br/>
> ├─fashion <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─main.py <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─neralnet.py <br/>
> └─tests <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─neralnet_test.py <br/>

Then you can run the main script for each section of our reports:

    ./main.py -r checkGrad
    ./main.py -r showTraining
    ./main.py -r showRegularization
    ./main.py -r showActivation
    ./main.py -r showNetworkTopology

Note that  checker.py will fail because the pickled baseline uses a different initialized weights.