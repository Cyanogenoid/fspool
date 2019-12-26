# Autoencoder experiments

This directory contains the experiments for the auto-encoder experiments and the classification experiment on noisy MNIST sets.
To run the experiments, you can use:

```
./polygon-experiment.sh

./noise-experiment.sh 1
./classify-experiment.sh 1
```

The first command runs the polygon experiment.
The last two commands run the MNIST auto-encoder experiments and classification experiments.
The argument for them specifies the run number; trained models are saved with this as suffix into `logs` so that they don't overwrite each other.
Note that the MNIST classification experiments must be run after the auto-encoder ones because some of them require pre-trained weights.
All of these shell scripts simply call `python train.py` with the appropriate arguments.
You can see the detailed help message for the available arguments with `python train.py -h`.

Once the classification models have been trained and stored into the logs directory, you can calculate the mean and standard deviation over multiple runs with `python summarise-mnist.py`.
To plot MNIST predictions of the trained FSPool-based auto-encoder and baseline model, you can run `python plot-mnist.py --resume logs/mnist-* --dim 32 --latent 16`.
To plot dataset samples, you can run `python plot-dataset.py`.

## File structure
The main training script is `train.py`.
This loads a dataset defined in `data.py` and trains a model defined in `model.py`.
The encoder uses FSPool (and FSUnpool with auto-encoders) defined in `fspool.py`.


## Requirements
- Python 3
- PyTorch 1.0+
- torchvision
- numpy
- scipy
- pandas
- matplotlib
- tqdm
