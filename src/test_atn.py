import torch
from models.losses import nce_loss_fn, InfoNCE
from models.TSCP2 import TSCP
from torch.functional import F
def eval_test(X1, X2):
    y = torch.rand(640, 1).cuda()
    res = []
    sim = F.cosine_similarity(torch.unsqueeze(X1, dim=1), 
                                torch.unsqueeze(X2, dim=0), dim=1)
    res.append(sim)
    res.append(y)
    d = torch.concat(res, 0)
    gt = torch.zeros(y.shape[0], 1).cuda()
    gt[torch.where((y > int(2 * 100 * 0.15)) & (y < int(2 * 100 * 0.85)))[0]] = 1
    from sklearn.metrics import confusion_matrix,f1_score
    gt_pred = y.detach().cpu().numpy()
    import numpy as np
    print(gt_pred.shape)
    gt_pred = np.where(gt_pred<0.5, 1, 0)
    print(f1_score(gt_pred, gt.detach().cpu().numpy()))
    print(y.shape)
    print(sim.shape)

def cosine(X1, X2):
    # criterion = InfoNCE(reduction='none')
    # loss1 = criterion(X1, X2)
    # # print(loss1, loss2)
    # loss2, mean_sim, mean_neg = criterion(X1, X2)
    # print(loss2, mean_sim, mean_neg)
    sim = F.cosine_similarity(torch.unsqueeze(X1, dim=1), 
                                torch.unsqueeze(X2, dim=0), dim=1)
    print(sim.shape)
    # sim = sim.reshape(-1, 1)
    y = torch.rand(64, 1).cuda()
    res = torch.concat((y, sim), 0)
    print(res.shape)

if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    
    X = torch.rand(64, 1, 100).cuda()

    
    model =  TSCP(input_size=1, 
                    hidden_dim=(100), 
                    output_size= 20, #Encoder part
                    num_channels= [1, 32, 64],
                    kernel_size= 4, 
                    dropout= 0, 
                    batch_norm=True,
                    attention=True, 
                    non_linear= 'relu')
    # tcn = TCN(in_channel= 1,
    #             out_channel= 64,
    #             num_dilations= [1, 2, 4, 8, 16],
    #             kernel_size= 4,
    #             dropout= 0,
    #             batch_norm= True,
    #             seq_length= 100,
    #             attention= False,
    #             non_linear= 'relu').cuda()
    
    x = model(X)
    

    
    # X1 = torch.rand(64, 10).cuda()
    # X2 = torch.rand(64, 10).cuda()
    
    