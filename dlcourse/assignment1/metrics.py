import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for (pred, true) in zip(prediction, ground_truth):
        if pred == True and true == True: tp += 1
        elif pred == True and true == False: fp += 1
        elif pred == False and true == False: tn += 1
        elif pred == False and true == True: fn += 1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    
    '''
    correct = 0
    for (pred, true) in zip(prediction, ground_truth):
        if pred == true:
            correct += 1
    return correct / prediction.shape[0]