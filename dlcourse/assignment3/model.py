import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """

        self.layers = [
            ConvolutionalLayer(in_channels=input_shape[2], out_channels=conv1_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(pool_size=4, stride=4),
            ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(pool_size=4, stride=4),
            Flattener(),
            FullyConnectedLayer(n_input=4*conv2_channels, n_output=n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # reset grads
        for layer in self.layers:
          for _, param in layer.params().items():
            param.grad = np.zeros_like(param.value)

        #forward
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        
        loss, d_out = softmax_with_cross_entropy(out, y)

        # backward
        for layer in self.layers[::-1]:
            d_out = layer.backward(d_out)

        return loss

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return np.argmax(X, axis=1)

    def params(self):
        result = {}

        names = ('Conv1', 'ReLU1', 'MaxPool1', 'Conv2', 'ReLU2', 'MaxPool2', 'Flattener', 'FC')

        for name, layer in zip(names, self.layers):
          for param_name, param_value in layer.params().items():
            result[name + '_' + param_name] = param_value

        return result
