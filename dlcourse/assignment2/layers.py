import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
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
      return -np.sum(np.log(probs[np.arange(len(target_index)), target_index.T]))
    
    return -np.sum(np.log(probs[target_index]))


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
    
    d_preds = probs - ground_truth

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