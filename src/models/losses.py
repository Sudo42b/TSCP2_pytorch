import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

import torch
import torch.nn as nn

def cosine_simililarity_dim1(x, y):
    return -F.cosine_similarity(x, y, dim=1)


def loss_fn(history, future, similarity, temperature=0.1):
    loss, pos, neg = nce_loss_fn(history, future, similarity, temperature)
    return loss, pos, neg

def nce_loss_fn(history, future,
                temperature=0.1,
                reduction='sum'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    N = history.shape[0]
    sim = -F.cosine_similarity(torch.unsqueeze(history, dim=1), 
                                torch.unsqueeze(future, dim=0), dim=2)
    
    pos_sim = torch.exp(torch.linalg.diagonal(sim)/temperature)

    tri_mask = np.ones(N ** 2, dtype= bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    tri_mask = torch.tensor(tri_mask).to(device)
    
    neg = torch.masked_select(sim, tri_mask).reshape(N, N - 1)
    all_sim = torch.exp(sim/temperature)
    
    logits = torch.divide(pos_sim.sum(), all_sim.sum(axis=1))
    lbl = torch.ones(history.shape[0]).to(device)
    # categorical cross entropy
    loss = F.binary_cross_entropy_with_logits(logits, lbl, reduction=reduction)
    loss = logits.sum()
    # divide by the size of batch
    loss = loss / lbl.shape[0]
    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.linalg.diagonal(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg


def dcl_loss_fn(history, future, similarity, temperature='0.1', debiased=True, tau_plus=0.1):
    # from Debiased Contrastive Learning paper: https://github.com/chingyaoc/DCL/
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # t : temperature scaling
    # tau_plus : class probability

    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = torch.exp(torch.linalg.diagonal(sim)/temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = torch.masked_select(sim, tri_mask).reshape(N, N-1)
    neg_sim = torch.exp(neg/temperature)

    # estimator g()
    if debiased:
        N = N-1
        Ng = (-tau_plus * N * pos_sim + torch.sum(neg_sim, axis=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(input= Ng, 
                         min= N * np.e ** (-1 / temperature), 
                         max= torch.finfo(torch.float32).max)
    else:
        Ng = torch.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = torch.mean(- torch.log(pos_sim / (pos_sim + Ng)))

    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.linalg.diagonal(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg


def fc_loss_fn(history, future, similarity, temperature=0.1, elimination_th = 0, elimination_topk = 0.1, attraction = False):
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    N = history.shape[0]
    if elimination_topk > 0.5:
        elimination_topk = 0.5
    elimination_topk = np.math.ceil(elimination_topk * N)

    sim = similarity(history, future)/temperature

    pos_sim = torch.exp(torch.linalg.diagonal(sim))

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg_sim = torch.masked_select(sim, tri_mask).reshape(N, N - 1)

    #sorted_ind = tf.argsort(neg_sim, axis=1)
    sorted_sim = torch.sort(neg_sim, axis=1)
    if elimination_th > 0:
        # Threshold-base cancellation only --TODO
        threshold = torch.tensor([elimination_th])
        mask =  torch.greater(threshold,sorted_sim).type(type(torch.float()))
        neg_count =  torch.sum(mask,axis=1)
        neg = torch.divide(torch.sum(sorted_sim * mask, axis=1),neg_count)
        neg_sim = torch.sum(torch.exp(sorted_sim/temperature) * mask, axis=1)

    else:
        # Top-K cancellation only
        if elimination_topk == 0:
            elimination_topk = 1
        tri_mask = np.ones(N * (N - 1), dtype=np.bool).reshape(N, N - 1)
        tri_mask[:, -elimination_topk:] = False
        neg = torch.masked_select(sorted_sim, tri_mask).reshape(N, N - elimination_topk - 1)
        neg_sim = torch.sum(torch.exp(neg), axis=1)

    #logits = tf.divide(K.sum(pos_sim, axis=-1), pos_sim+neg_sim)
    #lbl = np.ones(N)
    # categorical cross entropy
    #loss = criterion(y_pred = logits, y_true = lbl)
    loss = torch.mean(- torch.log(pos_sim / (pos_sim + neg_sim)))

    # divide by the size of batch
    # loss = loss / N
    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.linalg.diagonal(sim)) * temperature
    mean_neg = torch.mean(neg) * temperature
    return loss, mean_sim, mean_neg

def hard_loss_fn(history, future, similarity, temperature, beta = 0, debiased = True, tau_plus = 0.1):
    # from ICLR2021 paper: Contrastive LEarning with Hard Negative Samples https://www.groundai.com/project/contrastive-learning-with-hard-negative-samples
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # t : temperature scaling
    # tau_plus : class probability
    #
    # reweight = (beta * neg) / neg.mean()
    # Neg = max((-N * tau_plus * pos + reweight * neg).sum() / (1 - tau_plus), e ** (-1 / t))
    # hard_loss = -log(pos.sum() / (pos.sum() + Neg))
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    N = history.shape[0]
    history, future = normalize(history, future)
    sim = similarity(history, future)
    pos_sim = torch.exp(torch.linalg.diagonal(sim)/temperature)

    tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = torch.masked_select(sim, tri_mask).reshape(N,N-1)
    neg_sim = torch.exp(neg/temperature)

    reweight = (beta * neg_sim) / torch.mean(neg_sim, axis=1).reshape(-1, 1)
    if beta == 0:
        reweight = 1
    # estimator g()
    if debiased:
        N = N-1
        #(-N*tau_plus*pos + reweight*neg).sum() / (1-tau_plus)

        Ng = (-tau_plus * N * pos_sim + torch.sum(reweight * neg_sim, axis=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(input=Ng, 
                        min=np.e ** (-1 / temperature),
                        max=torch.finfo().max)
    else:
        Ng = torch.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = torch.mean(-torch.log(pos_sim / (pos_sim + Ng)))
    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.linalg.diagonal(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg



class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
    Input shape:
        history: (N, D) Tensor with history samples (e.g. embeddings of the input).
        future: (N, D) Tensor with future samples (e.g. embeddings of augmented input).
    Returns:
        Value of the InfoNCE Loss.
    Examples:
        >>> loss = InfoNCE()
        >>> batch_size, embedding_size = 64, 10
        >>> history = torch.randn(batch_size, embedding_size)
        >>> future = torch.randn(batch_size, embedding_size)
        >>> output = loss(history, future)
    """

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, history, future):
        return info_nce(history, future,
                        temperature=self.temperature,
                        reduction=self.reduction)

def info_nce(history, future, temperature=0.1, reduction='mean'):
    
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    # Check input dimensionality.
    if history.dim() != 2:
        raise ValueError('<history> must have 2 dimensions.')
    if future.dim() != 2:
        raise ValueError('<future> must have 2 dimensions.')

    # Check matching number of samples.
    if len(history) != len(future):
        raise ValueError('<history> and <future> must must have the same number of samples.')

    # Embedding vectors should have same number of components.
    if history.shape[-1] != future.shape[-1]:
        raise ValueError('Vectors of <history> and <future> should have the same number of components.')

    # Normalize to unit vectors
    history, future = normalize(history, future)
    
    N = history.shape[0]
    
    # Cosine Similarity
    sim = F.cosine_similarity(torch.unsqueeze(history, dim=1), 
                                torch.unsqueeze(future, dim=0), dim=2)
    all_sim = torch.exp(sim/temperature)
    
    # Negative Mask
    tri_mask = np.ones(N ** 2, dtype= bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tri_mask = torch.tensor(tri_mask).to(device)
    # N-1 mean except to positive
    neg = torch.masked_select(sim, tri_mask).reshape(N, N - 1)
    #Positive Similarity
    pos_sim = torch.exp(torch.linalg.diagonal(sim)/temperature)
    logits = torch.divide(pos_sim.sum(), all_sim.sum(axis=1))
    
    label = torch.ones(history.shape[0]).to(device)
    
    # categorical cross entropy
    loss = F.binary_cross_entropy_with_logits(logits, label, reduction=reduction)
    # loss = torch.sum(logits)
    # divide by the size of batch
    # loss = loss / lbl.shape[0]
    # similarity of positive pairs (only for debug)
    mean_sim = torch.mean(torch.linalg.diagonal(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg
    

