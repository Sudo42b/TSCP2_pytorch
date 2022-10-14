import torch
from models.losses import nce_loss_fn, InfoNCE
from models.TSCP2 import TSCP
from torch.functional import F
from models.resnet_tcn import RES_TCN
if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    
    X = torch.rand(64, 1, 700).cuda()
    # model = RES_TCN()
    
    model =  TSCP(input_size=(64, 100, 1), hidden_dim=(100//2), output_size= 10, #Encoder part
                tcn_out_channel=64, num_channels= [3, 5, 7],
                kernel_size= 4, dropout= 0.2, batch_norm=False,
                attention=True, non_linear= 'relu')
    
    x = model(X).cuda()
    print(x.shape)
    exit()
    X1 = torch.rand(64, 10).cuda()
    X2 = torch.rand(64, 10).cuda()
    
    
    # criterion = InfoNCE(reduction='none')
    # loss1 = criterion(X1, X2)
    # # print(loss1, loss2)
    # loss2, mean_sim, mean_neg = criterion(X1, X2)
    # print(loss2, mean_sim, mean_neg)
    # sim = F.cosine_similarity(torch.unsqueeze(X1, dim=1), 
    #                             torch.unsqueeze(X2, dim=0), dim=1)
    # sim = sim.reshape(-1, 1)
    # y = torch.rand(640, 1).cuda()
    # res = []
    # res.append(sim)
    # res.append(y)
    # d = torch.concat(res, 0)
    # gt = torch.zeros(y.shape[0], 1).cuda()
    # gt[torch.where((y > int(2 * 100 * 0.15)) & (y < int(2 * 100 * 0.85)))[0]] = 1
    # from sklearn.metrics import confusion_matrix,f1_score
    # gt_pred = y.detach().cpu().numpy()
    # import numpy as np
    # print(gt_pred.shape)
    # gt_pred = np.where(gt_pred<0.5, 1, 0)
    # print(f1_score(gt_pred, gt.detach().cpu().numpy()))
    # print(y.shape)
    # print(sim.shape)