import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

############ FocalLoss

def FocalLoss(y_pred,y_true, gamma=2, alpha=1):
    y_true=y_true.detach()
    bce_loss = F.binary_cross_entropy(y_pred, y_true.float())
    loss =alpha * (1 - torch.exp(-bce_loss)) ** gamma * bce_loss
    return loss
    
    

# test
model = nn.Linear(10,10)
x = Variable(torch.randn(1, 10).float(),requires_grad=True)
target = torch.randint(0,2,(10,)).float()
output = model(x)
sig=nn.Sigmoid()
output = sig(output)
loss=FocalLoss(output,target) 
loss.backward()
print(model.weight.grad)


############### Generalized dice loss
def GeneralizedDiceLoss(y_pred,y_true): 
    Nb_img = y_pred.shape[-1]
    r = torch.zeros((Nb_img,2))
    for l in range(Nb_img): r[l,0] = torch.sum(y_true[:,:,l]==0)
    for l in range(Nb_img): r[l,1] = torch.sum(y_true[:,:,l]==1)
    p = torch.zeros((Nb_img,2))
    for l in range(Nb_img): p[l,0] = torch.sum(y_pred[:,:,l][y_true[:,:,l]>0])
    for l in range(Nb_img): p[l,1] = torch.sum(y_pred[:,:,l][y_true[:,:,l]<0])
    
    w = torch.zeros((2,))
    w[0]=1/(torch.sum(r[:,0])**2)
    w[1]=1/(torch.sum(r[:,1])**2)
    
    num=(w[0]*torch.sum(r[:,0]*p[:,0]))+(w[1]*torch.sum(r[:,1]*p[:,1]))
    denom=(w[0]*torch.sum(r[:,0]+p[:,0]))+(w[1]*torch.sum(r[:,1]+p[:,1]))
    
    return 1-(2*(num/denom))
    
# test
x = Variable(torch.randn((10, 10, 3)).float(), requires_grad=True)
target = torch.randint(0, 2, (10, 10, 3)).float()
output = sig(x)
loss=GeneralizedDiceLoss(output,target) 
loss.backward()
print(model.weight.grad)

x = Variable(torch.randn((10, 10, 3)).float(), requires_grad=True)
output = sig(x)
loss=FocalLoss(output,target) 
loss.backward()
print(model.weight.grad)

















