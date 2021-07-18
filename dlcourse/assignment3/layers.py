import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W * W)
    grad = 2 * reg_strength * W
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if predictions.ndim > 1:
      fixed_predictions = predictions - np.max(predictions, axis=1).reshape(-1, 1)
      return np.exp(fixed_predictions) / np.sum(np.exp(fixed_predictions), axis=1).reshape(-1, 1)
    
    fixed_predictions = predictions - np.max(predictions)
    return np.exp(fixed_predictions) / np.sum(np.exp(fixed_predictions))


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim > 1:
      return -np.mean(np.log(probs[np.arange(len(target_index)), target_index.T]))
    
    return -np.mean(np.log(probs[target_index]))


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    ground_truth = np.zeros(preds.shape)
    
    if ground_truth.ndim > 1:
      ground_truth[np.arange(len(target_index)), target_index.T] = 1
    else:
      ground_truth[target_index] = 1
    
    d_preds = (probs - ground_truth) / preds.shape[0]

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
      pass

    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_relu = np.greater(self.X, 0)
        d_result = d_out * d_relu
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = X.dot(self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        dLdW = self.X.T.dot(d_out)
        dLdB = np.ones((1, d_out.shape[0])).dot(d_out)

        self.W.grad += dLdW
        self.B.grad += dLdB

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        stride, int - step of the filter
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = stride


    def forward(self, X):
        X_with_pad = np.zeros((X.shape[0], X.shape[1]+2*self.padding, X.shape[2]+2*self.padding, X.shape[3]))
        X_with_pad[:, self.padding:X.shape[1]+self.padding, self.padding:X.shape[2]+self.padding, :] = X
        self.X_with_pad = X_with_pad
        
        batch_size, height, width, _ = X.shape

        out_height = int((height - self.filter_size + 2 * self.padding) / self.stride + 1)
        out_width = int((width - self.filter_size + 2 * self.padding) / self.stride + 1)

        V = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W = self.W.value.reshape(-1, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_with_pad[:, y:y+self.filter_size, x:x+self.filter_size, :]
                X_slice = X_slice.reshape(batch_size, -1)

                V_slice = X_slice.dot(W) + self.B.value

                V[:, y, x, :] = V_slice
        
        return V


    def backward(self, d_out):
        X_with_pad = self.X_with_pad
        padding = self.padding

        batch_size, height, width, _ = X_with_pad.shape
        _, out_height, out_width, _ = d_out.shape


        d_result = np.zeros_like(X_with_pad)
        W = self.W.value.reshape(-1, self.out_channels)
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_with_pad[:, y:y+self.filter_size, x:x+self.filter_size, :]
                inp_shape = X_slice.shape
                X_slice = X_slice.reshape(batch_size, -1)

                d_out_slice = d_out[:, y, x, :]

                dLdW = X_slice.T.dot(d_out_slice).reshape(self.W.value.shape)
                dLdB = np.ones((1, d_out_slice.shape[0])).dot(d_out_slice).reshape(self.B.value.shape)

                self.W.grad += dLdW
                self.B.grad += dLdB

                d_result[:, y:y+self.filter_size, x:x+self.filter_size, :] += d_out_slice.dot(W.T).reshape(inp_shape)

        return d_result[:, padding:height-padding, padding:width-padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
