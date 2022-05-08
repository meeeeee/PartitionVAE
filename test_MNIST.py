import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms
import PIL
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import PVAE
import utils

h,w,f = 14,14,2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=128)

model = PVAE(encoder=[[5,32],[5,8]], partition=[[5,2],[10,4]], linlistdec=[100,h*w], convlistdec=[[5,1]]).to(device)
learning_rate,epochs,recon_factor,distro_factor,reprsize,noise_factor = 0.00025,25,100,10**(-np.log2(f)-3),6,0.01
optim = optimizer(model, learning_rate)
epoch_train_losses,epoch_test_losses = AverageMeter(),AverageMeter()

for epoch in range(epochs):
  model.train()
  with tqdm(total=(len(train_loader)),ncols=100) as train:
    train.set_description('Train Epoch: {}/{}'.format(epoch,epochs-1))
    for idx,data in enumerate(train_loader,0):
        batch, _ = data
        batch = torch.clamp((batch+noise_factor*torch.randn(batch.size())),min=0,max=1).to(device)
        x,mean,logvar = model(batch)
        loss = VAELoss(batch,x,mean,logvar,recon_factor,distro_factor)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_train_losses.update(loss.item(), len(batch))
        train.set_postfix(loss='{:.6f}'.format(epoch_train_losses.avg))
        train.update(1)
  model.eval()
  with torch.no_grad():
    with tqdm(total=len(test_loader),ncols=90) as test:
      test.set_description('Test Epoch:  {}/{}'.format(epoch,epochs-1))
      for idx,data in enumerate(test_loader,0):
        batch,_ = data
        batch = batch.to(device)
        x,mean,logvar = model(batch)
        loss = VAELoss(batch,x,mean,logvar,recon_factor,distro_factor)
        epoch_test_losses.update(loss.item(), len(batch))
        test.set_postfix(loss='{:.6f}'.format(epoch_test_losses.avg))
        test.update(1)

torch.save(model, "PVAE.pth")
model.eval()
# run tests on model by viewing output images and perturbing mean and log variance for some representation to see how the output changes
