<p align="left">
  <img width="300" height="80" src="./figures/full_logo.png">
</p>

# StarNet Self-Supervised

Predicting chemical abundances and parameters from stellar spectra using self-supervised learning

## Dependencies

- Python 3.9.6

- [PyTorch](http://pytorch.org/): `pip install torch==1.13.1`

- h5py: `pip install h5py`

## Data download

## Training the Network

### Option 1

1. The model architecture and hyper-parameters are set within configuration file in [the config directory](./configs). For instance, I have already created the [original configuration file](./configs/starnet_ss_1.ini). You can copy this file under a new name and change whichever parameters you choose.
  
2. If you were to create a config file called `starnet_ss_2.ini` in Step 1, this model could be trained by running `python train_starnet_ss.py starnet_ss_2 -v 5000 -ct 10.00` which will train your model displaying the progress every 5000 batch iterations and saves the model every 10 minutes. This same command will continue training the network if you already have the model saved in the [model directory](./models) from previous training iterations. 

### Option 2

Alternatively, if operating on compute-canada, you can use the `cc/launch_starnet_ss.py` script to simultaneously create a new configuration file and launch a bunch of jobs to train your model. 

1. Change the [load modules file](./cc/module_loads.txt) to include the lines necessary to load your own environment with pytorch, etc. 
2. Then, to copy the [original configuration](./configs/starnet_ss_1.ini), but use, say, a batch size of 16 spectra, you could use the command `python cc/launch_starnet_ss.py starnet_ss_2 -bs 16`. This will launch two 3-hour jobs on the GPU nodes to finish the training. You can checkout the other parameters that can be changed with the command `python cc/launch_starnet_ss.py -h`.

## Analysis notebooks

1. Checkout the [test notebook](./test_starnet_ss.ipynb) to evaluate the trained network.
