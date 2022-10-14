import sys
import numpy as np
import os

import matplotlib.pyplot as plt
# dataset
from models import TSCP2 as cp2
from models import losses as ls
from utils.dataloader import Load_Dataset
from utils.log_writer import Logger
#from utils.util import accuracy_fn
from tqdm import tqdm
from models.TSCP2 import TSCP
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# Train
import torch
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from torch.optim import Adam

#### Configuration
from hydra import compose, initialize
from omegaconf import DictConfig

# global initialization
with initialize(version_base=None, config_path='./config/'):
    args:DictConfig = compose(config_name="hyper.yaml")
    print('interface of running experiments for TSCP2 baselines')
    DATA_PATH = args.exp_param.data_path
    OUTPUT_DIR = args.exp_param.output_dir
    OUTPUT_PATH = os.path.join(args.exp_param.output_dir,args.exp_param.dataset)
    MODEL_PATH = os.path.join(args.exp_param.output_dir, "model")
    DATASET_NAME = args.exp_param.dataset
    LOSS = args.exp_param.loss
    SIM = args.exp_param.sim
    GPU = args.exp_param.gpu
    LOG_PATH = os.path.join(args.exp_param.output_dir,args.exp_param.dataset)
    
    # hyperparameters for grid search
    WINDOW_SIZE = args.hyper_param.window_size
    ENCODE_DIM = args.hyper_param.encode_feature
    BATCH_SIZE = args.hyper_param.batch_size
    EPOCHS = args.hyper_param.epoch
    LR = args.hyper_param.lr
    TEMP = args.hyper_param.temp_pram
    TAU = args.hyper_param.tau
    BETA = args.hyper_param.beta
    EVALFREQ = args.hyper_param.eval_freq
    decay_steps = args.hyper_param.decay_steps
    PATIENCE = args.hyper_param.patience


# Sanity check
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pred_sim"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "model"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, DATASET_NAME), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, DATASET_NAME, "plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, DATASET_NAME, "pred_sim"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, DATASET_NAME, "model"), exist_ok=True)

writer = SummaryWriter(log_dir=LOG_PATH)
from datetime import datetime
log_name = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")+'.log'



train_name = f"DS({DATASET_NAME})_T({TEMP})_WIN_SZ({WINDOW_SIZE})_BS({BATCH_SIZE})\
                _DIM({ENCODE_DIM})_lr({LR})__LOSS({LOSS})_SIM({SIM})\
                _TAU({TAU})_BETA({BETA})".replace(' ', '')
print("------------------------------------>>> " + train_name)


# -------------------------------
# 1 PREPARE DATASET
# -------------------------------
train_ds = Load_Dataset(DATA_PATH, 
                        DATASET_NAME, 
                        WINDOW_SIZE, 
                        BATCH_SIZE, mode = "train")
val_ds = Load_Dataset(DATA_PATH, 
                        DATASET_NAME, 
                        WINDOW_SIZE, 
                        BATCH_SIZE, mode = "test")

# ------------------------
# 2 TRAINING
# ------------------------
# Model Load!
from models.resnet1d import ResNet1D, BasicBlock1D
# model = ResNet1D(BasicBlock1D, [2, 2, 2, 2])
model = TSCP(input_size= (BATCH_SIZE, WINDOW_SIZE, 1), 
                    hidden_dim= (WINDOW_SIZE//2), 
                    output_size= ENCODE_DIM, #Encoder part
                    tcn_out_channel=64, 
                    num_channels= [3, 5, 7],
                    kernel_size= 4, dropout= 0.2, batch_norm=False,
                    attention=False, 
                    non_linear= 'relu')


DEVICE = torch.device(GPU if torch.cuda.is_available() else 'cpu')

#### DATA PARALLEL START ####
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
    
optimizer = Adam(params=model.parameters(), lr=LR)

lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, 
                                            T_0= len(train_ds)//BATCH_SIZE,
                                            T_mult= 1,
                                            gamma= 0.5,
                                            eta_max=LR)

from models.losses import InfoNCE
from models.losses import nce_loss_fn
criterion = InfoNCE(temperature=TEMP, reduction='mean')
# criterion = nce_loss_fn
#BCEWithLogitsLoss(reduction='sum')
if SIM == "cosine":
    # ls.cosine_simililarity_dim1
    similarity = ls.cosine_simililarity_dim1
from utils.log_writer import Logger
from utils.trainer import Trainer

# Train
trainer = Trainer(train_set= train_ds, 
                    val_set= val_ds, 
                    test_set= val_ds, 
                    model= model,
                    optimizer= optimizer, 
                    scheduler= lr_scheduler, 
                    num_classes = 1,
                    loss_fn= criterion,
                    patience= PATIENCE, 
                    writer=writer, 
                    save_path=os.path.join(MODEL_PATH, f"{train_name}.pth"),
                    device= DEVICE)

train_log, val_log = trainer.train_loop(epochs= EPOCHS,
                        win_sz= WINDOW_SIZE,
                        temp= 0.1, 
                        beta= BETA, 
                        tau= TAU)

# SAVE MODEL and Learning Progress plot
#with plt.xkcd():
splot=1
if splot ==1:
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(train_name)
    ax1.plot(train_log['train_loss'], label="Loss")
    ax2.plot(train_log['train_sim'], label="Positive pairs")

    ax2.plot(train_log['train_neg'], label="Negative pairs")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH,"plots", LOSS+"__"+train_name + "_LOSS.png"))
    print("Learning progress plot saved!")

# -------------------------
# 4 TEST SET & SEGMENTATION
# -------------------------
from utils.metric import estimate_CPs
model.eval()
log, gt, y_sim = trainer.test_loop(win_sz=WINDOW_SIZE)
result = estimate_CPs(sim=y_sim, gt=gt, threshold=0.5)

np.savetxt(os.path.join(OUTPUT_PATH, "pred_sim", train_name + "_pred_sim.csv"), 
            np.concatenate((gt, np.array(y_sim).reshape((y_sim.shape[0],1))),1),
            delimiter=',',
            header="lbl,"+LOSS, comments="")
print("Saved test similarity result!")


print('Average similarity for test set : Reps : {}'.format(np.mean(y_sim)))

with open(os.path.join(OUTPUT_PATH, "Evaluation2.txt"), "a") as out_file:
    out_file.write(f"{BATCH_SIZE}, {WINDOW_SIZE}, {ENCODE_DIM},{optimizer.state_dict()},\
                    {TEMP}, {LR}, {np.mean(train_log['train_loss'])}, {train_log['train_sim'][-1]}, \
                    {train_log['train_neg'][-1]}, {result}")
    print("Saved model to disk")

# -------------------------
# 3 SAVE THE MODEL
# -------------------------

import torch
writer.close()
torch.save(model.state_dict(), os.path.join(MODEL_PATH, train_name + ".pth"))
print("Saved model to disk")
import json

with open(os.path.join(MODEL_PATH, train_name + ".json"), 'w') as outfile:
    #json_file.write(model_json)
    json.dump(model, outfile, indent=4, sort_keys=True)
    print("Saved model to disk")
