import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import PIL
from tqdm import tqdm

class PVAE(nn.Module):
  """
  encoder, convlistdec: list of conv layers as lists/tuples with first entry kernel and second entry channels
  linlistdec: list of linear layer sizes
  partition: partition of representation layer as list of lists of Linear layer sizes
  """
  def __init__(self, encoder=[[5,32],[5,8]], partition=[[5,2],[10,4]], linlistdec=[100,100], convlistdec=[[5,1]]):
    super(PVAE, self).__init__()
    encoderlist,decoderlinlist,decoderconvlist = [],[],[]
    convshape, channels = np.array([f*h,f*w]), 1
    for auto in encoder:
      encoderlist.append(nn.Conv2d(in_channels=channels,out_channels=auto[1],kernel_size=auto[0],stride=1,padding='valid').to(device))
      encoderlist.append(nn.ReLU().to(device))
      encoderlist.append(nn.MaxPool2d(kernel_size=2).to(device))
      channels = auto[1]
      convshape = ((convshape-auto[0]+1)/2).astype(int)
    encoderlist.append(nn.Flatten().to(device))
    length = convshape[0]*convshape[1]*channels

    if 0 in [sum(auto) for auto in partition]: raise Exception("empty layer description in partition")
    self.partitions_mean, self.partitions_logvar = [], []
    self.split = []
    for auto in partition:
      lst = []
      partlen = auto[-1]#length
      self.split.append(auto[-1])
      for x in auto:
        lst.append(nn.Linear(in_features=partlen,out_features=x).to(device))
        partlen = x
      lst.append(nn.Linear(in_features=partlen,out_features=partlen).to(device))
      self.partitions_mean.append(nn.Sequential(*lst).to(device))
      self.partitions_logvar.append(nn.Sequential(*lst).to(device))
    
    encoderlist.append(nn.Linear(in_features=length,out_features=sum(self.split)).to(device))
    self.encoder = nn.Sequential(*encoderlist).to(device)

    length = sum([auto[-1] for auto in partition])
    
    if linlistdec[-1] != h*w: raise Exception("last layer of linlistdec does not equal h*w")
    for auto in linlistdec:
      decoderlinlist.append(nn.Linear(in_features=length, out_features=auto).to(device))
      length=auto
    self.lineardecoder = nn.Sequential(*decoderlinlist).to(device)

    convshape,channels = np.array([h,w]), 1
    for auto in convlistdec:
      decoderconvlist.append(nn.ConvTranspose2d(in_channels=channels,out_channels=auto[1],kernel_size=auto[0],stride=1,padding=0).to(device))
      decoderconvlist.append(nn.ReLU().to(device))
      channels = auto[1]
      decoderconvlist.append(nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=auto[0],stride=1,padding='valid').to(device))
      decoderconvlist.append(nn.ReLU().to(device))
    decoderconvlist.append(nn.Conv2d(in_channels=channels,out_channels=1,kernel_size=1,stride=1,padding='valid').to(device))
    self.convdecoder = nn.Sequential(*decoderconvlist).to(device)

  def normalsample(self, mean, logvar):
    std = torch.exp(logvar/2).to(device)
    eps = torch.randn_like(std).to(device)
    return mean+std*eps

  def forward(self, x):
    x = self.encoder(x) # encoder
    x = torch.split(x, self.split, dim=-1)
    xmean,xlogvar = [self.partitions_mean[a](x[a]) for a in range(len(self.split))],[self.partitions_logvar[a](x[a]) for a in range(len(self.split))] # pass partitioned units through Linear layers
    xmean,xlogvar = torch.cat(xmean,-1),torch.cat(xlogvar,-1) # merge partitioned units --- prerepresentation layer
    x = self.normalsample(xmean,xlogvar) # sample normally given mean and logvar
    x = self.lineardecoder(x) # linear layers
    x = x.view([x.shape[0],1,h,w]) # reshape for deconv
    x = self.convdecoder(x) # deconv layers, should maintain shape
    x = nn.functional.interpolate(x,scale_factor=f,mode='nearest-exact') # upsample to yield same size as input image
    return x,xmean,xlogvar # return output image and distribution parameters

  def __getitem__(self, args):
    mean,logvar = args
    y = self.normalsample(mean,logvar)
    y = self.lineardecoder(y)
    y = y.view([y.shape[0],1,h,w])
    y = self.convdecoder(y)
    y = nn.functional.interpolate(y,scale_factor=f,mode='nearest-exact')
    return y
