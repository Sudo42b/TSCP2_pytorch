import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, \
                            classification_report, \
                            precision_recall_fscore_support
from matplotlib import pyplot as plt
import torch

# Accuracy
@torch.no_grad()
def accuracy_fn(y_pred, y_true):
    n_correct = torch.eq(y_pred, y_true).sum().item()
    
    accuracy = (n_correct / (y_pred.view(-1, 1).shape[0]))
    # accuracy = dice_coeff(y_true, y_pred)
    f1, precision, recall = f1_score(y_true=y_true, y_pred=y_pred)
    
    return accuracy, precision, recall, f1

@torch.no_grad()
def f1_score(y_true:torch.Tensor, 
            y_pred:torch.Tensor, 
            is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.item(), precision.item(), recall.item()

@torch.no_grad()
def model_performance(ground, pred, th=0.5):
    """ Custom way to quantify the performance of the peak detector"""
    
    TC, TP, FP, TC_p = 0, 0, 0, 0
    idx = np.arange(len(pred)-1)
    
    diff = np.diff(ground)
    a = diff > 0
    b = diff < 0

    if ground[0] == 1: a[0] = True
    if ground[-1] == 1: b[-1] = True
    if np.sum(a) > 0: 
        if np.sum(a) == np.sum(b):
            for l,n in zip(idx[a],idx[b]):
                if np.sum(pred[l:n]) / (n - l) > th:
                    TP += 1
                TC += 1
        else:
            print('Error during model performance check')
    
    diff = np.diff(pred)
    c = diff > 0
    d = diff < 0
    
    if pred[0] == 1: c[0] = True
    if pred[-1] == 1: d[-1] = True
    if np.sum(c) > 0:
        if np.sum(c) == np.sum(d):
            for l,n in zip(idx[c],idx[d]):
                if np.sum(ground[l:n]) / (n - l) < th:
                    FP += 1
                TC_p += 1
        else:
            print('Error during model performance check')        
            
    return TP, TC, FP, TC_p
from typing import Dict, List
def plot_loss_graph(train_log:Dict[List],
                    val_log:Dict[List], 
                    save_path=None):
    "train_loss"
    'train_acc'
    'train_pre'
    'train_rec'
    'train_f1'
    'val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_f1'
    # Plot performance
    plt.figure(figsize=(15,5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_log['train_loss'], label="train_loss")
    plt.plot(val_log['val_loss'], label="val_loss")
    plt.legend(loc='upper right')

    # Plot Metric
    plt.subplot(1, 2, 2)
    plt.title("Metric")
    plt.plot(train_log['train_acc'], label="train_acc")
    plt.plot(train_log['train_pre'], label="train_pre")
    plt.plot(train_log['train_rec'], label="train_rec")
    plt.plot('train_f1', label="train_f1")
    plt.plot('val_acc', label="val_acc")
    plt.plot('val_pre', label="val_pre")
    plt.plot('val_rec', label="val_rec")
    plt.plot('val_f1', label="val_f1")
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path)
    # Show plots
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    
    plt.rcParams["figure.figsize"] = (7,7)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_label_position('bottom') 
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j]*100:.1f}%)",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    print (classification_report(y_true, y_pred))
    # Display
    plt.show()

def get_performance(y_true, y_pred, classes):
    """Per-class performance metrics. """
    performance = {'overall': {}, 'class': {}}
    metrics = precision_recall_fscore_support(y_true, y_pred)

    # Overall performance
    performance['overall']['precision'] = np.mean(metrics[0])
    performance['overall']['recall'] = np.mean(metrics[1])
    performance['overall']['f1'] = np.mean(metrics[2])
    performance['overall']['num_samples'] = np.float64(np.sum(metrics[3]))

    # Per-class performance
    for i in range(len(classes)):
        performance['class'][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i])}
    return performance

def get_probability_distribution(y_prob, classes):
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results
