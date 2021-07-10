def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct = 0
    for (pred, true) in zip(prediction, ground_truth):
        if pred == true:
            correct += 1
    
    return correct / prediction.shape[0]
