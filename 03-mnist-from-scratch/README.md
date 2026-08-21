# MNIST Digit Classifier — From Scratch in NumPy

Third project — first neural network, implemented with raw NumPy instead of a deep learning framework. The goal wasn't the model, it was understanding exactly what a framework like PyTorch does under the hood before using one.

## Goal

Classify handwritten digits (0-9) using a 2-layer neural network, implementing the forward pass, backpropagation, and gradient descent by hand — no `.fit()`, no autograd.

## Dataset

MNIST, 70,000 images, 28x28 grayscale, 10 classes. Split 70/15/15 into train/validation/test.

## Architecture

- Input: 784 nodes (flattened 28×28 image)
- Hidden layer: 128 nodes, ReLU activation
- Output layer: 10 nodes, softmax activation
- Loss: cross-entropy

## What I did

- Preprocessed the data — normalized pixel values to [0, 1], one-hot encoded the labels.
- Initialized weights small and random, biases at zero.
- Implemented forward pass, softmax, cross-entropy loss, and backpropagation manually — including the simplified softmax + cross-entropy gradient (predicted - actual) and the ReLU derivative using a mask on the pre-activation values.
- Trained with mini-batch gradient descent(batch size 64) rather than full-batch or pure SGD — enough updates per epoch to train efficiently, smooth enough to sanity-check convergence.
- Tracked training loss per epoch and validation loss/accuracy every 5 epochs, and plotted the loss curve.

## Results

Final run (20 epochs, learning rate 0.1, properly averaged gradients):

| Metric | Value |
|---|---|
| Final training loss | 0.031 |
| Validation accuracy | 97.4% |
| Test accuracy | 97.4% |

## What I learned

Gradient averaging changes your effective learning rate. My first working version summed gradients over the batch instead of averaging them, which meant my learning rate of 0.01 was effectively being multiplied by the batch size (64) — that's why it converged so fast (97.6% accuracy in just 10 epochs). After switching to properly averaged gradients (dividing by batch size, the textbook-correct approach), the same learning rate became far too small, and training slowed down dramatically. Retuning the learning rate up to 0.1 recovered the fast convergence — but now in a way that's correctly batch-size-independent, so it won't silently break if I change the batch size later. Seeing this tradeoff play out directly, rather than just being told about it, made the relationship between learning rate and gradient scale click in a way reading about it hadn't.


## Tools

Python, NumPy, Pandas, scikit-learn, Matplotlib
