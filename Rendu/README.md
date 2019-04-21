# Simple architecture to compare 2 handwritten digits from MNIST
This project proposes a pytorch architecture to compare a pair of digit. The data is taken from the MNIST dataset, which is composed of images of handwritten digits (0-9).
The output of this model is 0 when the first digit is lesser or equal to the second and 1 otherwise.

## Baseline & 2-models architecture
Two diffrent architectures are used to perform this task. The first one, referred here as baseline, is a simple classification model that compute the probability of each image to belong to the 10 classes. The final result is obtained by performing a "manual" comparison using *argmax*. The second architecture uses the same classification, but uses as well a second model (hence the name 2-models architecture) that takes as input the output of the first model. It calculate the probability for the first digit to be smaller/equal or greater.

As many different models were used to obtain optimal results, only a few were retained and are therefore proposed in this project. The model for classification, acquired through the litterature cannot be changed. However, 2 models are proposed for the comparison: a small one (3 fully connected layers) and a big one (5 fully connected layers).

## Parameters
Different parameters can be tuned in this program. Here is a list of the different parameters and their default value:

* Learning rate : 1e-3
* Number of epochs : 25
* Number of runs : 10
* Comparison model: small one
* Weight sharing : yes
* Display baseline : no
* Display loses : no

## How to use
Place yourself in the directory and type *pyton test.py -h* to get all the different parameters and how to use them. Here is a little example with no weight sharing and with losses display:

```$python test.py -w -l```