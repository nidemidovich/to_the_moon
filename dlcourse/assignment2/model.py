import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)
        self.reg = reg

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        # reset grads
        for layer in (self.first_layer, self.second_layer):
          for _, param in layer.params().items():
            param.grad = np.zeros_like(param.value)
        
        # forward
        first_out = self.first_layer.forward(X)
        relu_out = self.relu.forward(first_out)
        second_out = self.second_layer.forward(relu_out)
        
        loss, d_out = softmax_with_cross_entropy(second_out, y)

        # backward
        d_out_second = self.second_layer.backward(d_out)
        d_out_relu = self.relu.backward(d_out_second)
        d_out_first = self.first_layer.backward(d_out_relu)

        # regularization
        for _, param in self.params().items():
          loss_reg, grad_reg = l2_regularization(param.value, self.reg)
          loss += loss_reg
          param.grad += grad_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        
        for layer in (self.first_layer, self.relu, self.second_layer):
          X = layer.forward(X)

        pred = np.argmax(X, axis=1)

        return pred

    def params(self):
        result = {}

        names = ('first', 'relu', 'second')
        layers = (self.first_layer, self.relu, self.second_layer)

        for name, layer in zip(names, layers):
          for param_name, param_value in layer.params().items():
            result[name + '_' + param_name] = param_value

        return result
