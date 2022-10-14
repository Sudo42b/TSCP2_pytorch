import os
from tqdm import tqdm
from models.tcn import TCN
import numpy as np

from utils.usc_ds_helper import ts_samples
#from losses import loss_fn
def train_step(xis, xjs, amodel, optimizer, 
            criterion, temperature, 
            sfn, lfn, beta, tau):
    # print("---------",xis.shape)
    with tf.GradientTape() as tape:
        zis = amodel(xis)
        zjs = amodel(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        #loss, mean_sim = ls.dcl_loss_fn(zis, zjs, temperature, lfn)
        loss, mean_sim, neg_sim = loss_fn(zis, zjs, similarity = sfn, loss_fn = lfn, 
                                            temperature=temperature, tau = tau, beta = beta, 
                                            elimination_th = 0, attraction = False)

    gradients = tape.gradient(loss, amodel.trainable_variables)
    optimizer.apply_gradients(zip(gradients, 
                                amodel.trainable_variables))

    return loss, mean_sim, neg_sim

import torch
def train_prep(model, dataset, outpath, optimizer, criterion, train_name, 
            win, temperature=0.1, epochs=100, sfn="cosine", lfn='nce', beta=0.1, tau=0.1):
    beta_curr = beta
    epoch_wise_loss = []
    epoch_wise_sim = []
    epoch_wise_neg = []
    end_condition = 0
    for epoch in tqdm(range(epochs)):
        counter = 0
        step_wise_loss = []
        step_wise_sim = []
        step_wise_neg = []
        for mbatch in dataset:
            counter += 1
            a, b, lbl = ts_samples(mbatch, win)

            # a = data_augmentation(mbatch)
            # b = data_augmentation(mbatch)

            loss, sim, neg = train_step(torch.unsqueeze(a, axis=2), torch.unsqueeze(b, axis=2), 
                                        model, optimizer, criterion,
                                        temperature, sfn, lfn, beta=beta_curr, tau=tau)
            step_wise_loss.append(loss)
            step_wise_sim.append(sim)
            step_wise_neg.append(neg)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        epoch_wise_sim.append(np.mean(step_wise_sim))
        epoch_wise_neg.append(np.mean(step_wise_neg))
        # wandb.log({"nt_INFONCEloss": np.mean(step_wise_loss)})
        # wandb.log({"nt_sim": np.mean(step_wise_sim)})
        #if epoch % (np.floor(epoch / 10)) == 0:
        #    beta_curr = beta_curr - (beta/10)
        if epoch % 1 == 0:
            result = "epoch: {} (step:{}) -loss: {:.3f} - avg rep sim : {:.3f} - avg rep neg : {:.3f}\n".\
            format(epoch + 1, counter, np.mean(step_wise_loss), np.mean(step_wise_sim), np.mean(step_wise_neg))
            with open(os.path.join(outpath, train_name + ".txt"), "a") as myfile:
                myfile.write("{:.4f},{:.4f},{:.4f}\n".format(np.mean(step_wise_loss),
                                                            np.mean(step_wise_sim),
                                                            np.mean(step_wise_neg)))
            print(result)
        if epoch > 5:
            if np.abs(epoch_wise_loss[-1] - epoch_wise_loss[-2]) < 0.0001 or epoch_wise_loss[-2] < epoch_wise_loss[-1]:
                end_condition += 1
            else:
                end_condition = 0
            if end_condition == 4:
                return epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, model

    return epoch_wise_loss, epoch_wise_sim, epoch_wise_neg, model
from torch import nn
from typing import List, Tuple


class Flatten(nn.Module):
    def __init__(self, window_size):
        super(Flatten, self).__init__()
        self.win_sz = window_size

    def forward(self, x):
        bs, ws = x.shape[:2]
        x = x.reshape(bs, bs*ws)
        
        return x 

class TSCP(nn.Module):
    def __init__(self,
                #Prediction/Projection Head
                input_size:Tuple[int], hidden_dim:int, output_size:int, 
                #Encoder part
                tcn_out_channel=64, num_channels:List[int]= [3, 5, 7],
                kernel_size= 4, dropout= 0.2, batch_norm=False,
                attention=False, non_linear= 'relu'):
        super(TSCP, self).__init__()
        bs, ws, ch = input_size
        self.window_size = ws
        self.tcn = TCN(in_channel= ch,
                    out_channel= tcn_out_channel,
                    num_channels= num_channels,
                    kernel_size= kernel_size,
                    dropout= dropout,
                    batch_norm= batch_norm,
                    max_length= ws,
                    attention= attention,
                    non_linear= non_linear)
        #bs, ws, 64
        self.out_place = output_size
        
        self.encoder = nn.Sequential(*[
            Flatten(self.window_size),
            nn.Linear(bs*ws,
                    2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim,
                    hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                    self.out_place)
            ])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        x = self.tcn(x)
        x = self.encoder(x)
        return x
