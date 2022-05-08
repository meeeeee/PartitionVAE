import torch
import torch.nn as nn

class AverageMeter(object): # useful tool I found in FSRCNN code
  def __init__(self):
    self.reset()

  def reset(self):
    self.val,self.avg,self.sum,self.count = 0,0,0,0

  def update(self,val,n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count

def VAELoss(x,x_approx,xmean,xlogvar,recon_factor,distro_factor):
  KLDiv = distro_factor*torch.sum(-1 - xlogvar + xmean**2 + xlogvar.exp()) # found this formula for K-L divergence online, idk how accurate it is
  return recon_factor*nn.MSELoss()(x_approx,x)+KLDiv

def optimizer(model, learning_rate):
  return torch.optim.Adam(model.parameters(), lr=learning_rate)
