# hyperspec-unmixing-conv-ae
This repository contains the pytorch implementation Hyperspectral Unmixing Using a Convolutional Neural Network Autoencoder.

# Dependencies
* __PyTorch 1.8.0__
* __Python 3.7.10__

# Quick Start
## Data
The datasets used in the code are publicly available and can be found [here](https://rslab.ut.ac.ir/data).<br>
Download the Samson dataset from the above-mentioned source. Follow the directory tree given below:<br>
```
|-- [root] hyperspec-unmixing-conv-ae\
    |-- [DIR] data\
        |-- [DIR] Samson\
             |-- [DIR] Data_Matlab\
                 |-- samson_1.mat
             |-- [DIR] GroundTruth
                 |-- end3.mat
                 |-- end3_Abundances.fig
                 |-- end3_Materials.fig
```

## Training
The shell script that trains the model (```samson_train.sh```) can be found in the [run folder](https://github.com/16sumanrana/hyperspec-unmixing-conv-ae/blob/master/run). You can simply alter the hyperparameters and other related model options in this script and run it on the terminal.<br>

## Abundance Map Extraction
The shell script that extracts the abundance maps and end-members (```extract.sh```) can be found in the [run folder](https://github.com/16sumanrana/hyperspec-unmixing-conv-ae/blob/master/run). Ensure that the charateristics of the model match exactly with the pre-trained version to be used for extraction.<br>

# Results
The following are the results of a deep autoencoder, Configuration Name: LReLU (see paper). You can experiment with other configurations by altering the command line arguments during model training.

## Abundance Maps
![abundances](https://github.com/16sumanrana/hyperspec-unmixing-conv-ae/blob/master/imgs/abundances.png)
**Left**: Tree, **Middle**: Water and **Right**: Rock.
