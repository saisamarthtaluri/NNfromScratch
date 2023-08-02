# Handwritten Digit Classification using Neural Network Implemented from Scratch

This repository contains a notebook for building a basic feedforward neural network from scratch, using only low-level libraries such as Numpy and Pandas, to classify the handwritten digits (0-9) from the MNIST dataset. The purpose of this implementation is to demonstrate the inner workings of a neural network, as well as provide a base for anyone looking to understand neural networks at a more fundamental level without the abstraction of high-level libraries like TensorFlow or PyTorch.

## Dependencies
This project is implemented in Python and requires *only* the following Python libraries:

* Numpy
* Pandas
* Matplotlib

## Code Breakdown
### Data Loading and Preprocessing
The load_data function loads the MNIST dataset, normalizes the pixel intensities, and randomly splits the data into training and validation sets.

### Neural Network Model
The model consists of multiple hidden layers, each using a ReLU activation function, and an output softmax layer. The weights and biases of the network are initialized using the Xavier initialization method in the initialize_parameters function.

### Forward Propagation
The forward_propagation function performs forward propagation, calculating the activation of each layer based on the input data and the current weights and biases.

### Backward Propagation
The backward_propagation function performs backward propagation, calculating the gradients with respect to the cost function for the weights and biases.

### Parameter Update
The update_parameters function updates the parameters using the gradients computed in the backward propagation step and a predefined learning rate.

### Model Training
The train_model function trains the model using the forward propagation, backward propagation, and parameter update functions. The model's accuracy on the training and validation set is printed every 100 epochs.

### Prediction
The predict function uses the trained model parameters to make predictions on new, unseen data.

### Visualization
The display_samples function uses Matplotlib to display a random selection of the test images along with their true and predicted labels.

## Usage
1. Clone the repository.
2. Download train.csv from https://www.kaggle.com/competitions/digit-recognizer/data and save it in the same directory.
3. Run all the cells in MNISTfromScratch.ipynb notebook.
   
## Output
The script outputs the training and validation accuracy for each 100th epoch. It also outputs the predicted labels for the test set, and visualizes a few test images with their true and predicted labels.

## Disclaimer
This code is for educational purposes and might not be optimized for production use. The model does not use any regularization, which can lead to overfitting if the number of training epochs is too high.

## Future Improvements
Future improvements could include adding regularization techniques like dropout or L2 regularization, implementing early stopping to prevent overfitting, or using a more efficient optimization algorithm. Additionally, the model can be expanded to include more layers or different activation functions.


