from operator import xor
from turtle import forward
import torch
import torch.nn as nn
from .network.network_unet import UNetRes
from .network.dncnn import DnCNN
from .network.test_utils import test_mode
import numpy as np
from torch.nn.functional import pad
# from torchsummary import summary

class NNclass(nn.Module):
    def __init__(self,numInChan=3,numOutChan=3,network='unet',train_network=True,sigma_map=True,sigma_factor=1.0,train_sigma_factor=True):
        super(NNclass,self).__init__()
        if network=='unet':
            self.network=UNetRes(numInChan+1 if sigma_map else numInChan,numOutChan,nc=[64, 128, 256, 512], nb=2, act_mode='E', downsample_mode="strideconv", upsample_mode="convtranspose")
        elif network=='dncnn':
            self.network=DnCNN(depth=12, in_channels=numInChan+1 if sigma_map else numInChan, out_channels=numOutChan, init_features=64, kernel_size=3)
        for p in self.network.parameters():
            p.requires_grad = train_network
        self.sigma_map=sigma_map
        self.sigma_factor=nn.parameter.Parameter(torch.tensor([sigma_factor]),requires_grad=train_sigma_factor)
        # summary(self.network, ((numInChan+1 if sigma_map else numInChan), 256, 256), device="cpu") 
        print(self.network)
    def preForward(self,x,**kwargs):
        return x
    def postForward(self,x,input,**kwargs):
        return x
    def forward(self,x,sigma,**kwargs):
        xH=x.size(2)
        xW=x.size(3)
        if xH % 8 != 0 or xW % 8 != 0:
            padH=int(np.ceil(float(xH)/8.0)*8-xH)
            padW=int(np.ceil(float(xW)/8.0)*8-xW)
            x=pad(x,(0,padW,0,padH),'reflect')
        # print("**************34*****************",x.shape) 
        x.requires_grad_()
        if self.sigma_map:
            noise_level_map = torch.tensor(sigma,dtype=x.dtype,device=x.device).expand(x.size(0),1,x.size(2),x.size(3))*self.sigma_factor
            x_sigma = torch.cat((x, noise_level_map), 1)
        else:
            x_sigma=x
        h=self.network(self.preForward(x_sigma,**kwargs))
        out = self.postForward(h,x,**kwargs)
        
        if xH % 8 != 0 or xW % 8 != 0:
            out = out[:,:,:xH,:xW]
        return out

class DPIRNNclass(NNclass):
    def __init__(self, numInChan=3, numOutChan=3, network='unet', train_network=True,sigma_map=True,sigma_factor=1.0,train_sigma_factor=True):
        super(DPIRNNclass, self).__init__(numInChan, numOutChan, network, train_network,sigma_map=sigma_map,sigma_factor=sigma_factor,train_sigma_factor=train_sigma_factor)
        if network=='unet':
            self.network=UNetRes(numInChan+1 if sigma_map else numInChan,numOutChan,nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        elif network=='dncnn':
            self.network=DnCNN(depth=12, in_channels=numInChan+1 if sigma_map else numInChan, out_channels=numOutChan, init_features=64, kernel_size=3)
        for p in self.network.parameters():
            p.requires_grad = train_network
    def calculate_grad(self, x, sigma):
        '''
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :param sigma: Denoiser level (std)
        :return: Dg(x), DRUNet output N(x)
        '''
        xH=x.size(2)
        xW=x.size(3)
        if xH % 8 != 0 or xW % 8 != 0:
            padH=int(np.ceil(float(xH)/8.0)*8-xH)
            padW=int(np.ceil(float(xW)/8.0)*8-xW)
            x=pad(x,(0,padW,0,padH),'reflect')
        x = x.float()
        x = x.requires_grad_()
        if self.sigma_map:
            noise_level_map = torch.tensor(sigma,dtype=x.dtype,device=x.device).expand(x.size(0),1,x.size(2),x.size(3))*self.sigma_factor
            x_sigma = torch.cat((x, noise_level_map), 1)
        else:
            x_sigma=x
        N=self.network(x_sigma)
        Dg=x-N
        if xH % 8 != 0 or xW % 8 != 0:
            Dg = Dg[:,:,:xH,:xW]
            N = N[:,:,:xH,:xW]
        return Dg, N