import math
import numpy as np
import torch
import torch.nn as nn

from torchinfo import summary




class vgg_block(nn.Module):
    
    def __init__(self, filters, kernel_size, pool_size=2 ,stride=1,
                 padding='same', batchnorm=True, nonlin=nn.ReLU()):
        
            super(vgg_block, self).__init__()
                
            in_filters, out_filters = filters
            block = [nn.Conv2d(in_filters, out_filters, kernel_size = kernel_size, stride=stride, padding=padding),
                     nn.BatchNorm2d(out_filters) if batchnorm else None,
                     nonlin,
                     nn.Conv2d(out_filters, out_filters, kernel_size = kernel_size, stride=stride, padding=padding),
                     nn.BatchNorm2d(out_filters) if batchnorm else None,
                     nonlin,
                     nn.MaxPool2d(pool_size,pool_size)              
                     ]
            # remove None
            block = [i for i in block if i is not None]            
            self.block = nn.Sequential(*block)
            
    def forward(self, x):
        
        return self.block(x) 
            
            



class vgg_encoder(nn.Module):
    
    
    def __init__(self, z_dim=64, filter_list=[(5,64),(64,80),(80,96)], kernel_list=[5,3,3], pool_list=2, stride_list=1, padding='same', batchnorm=True, nonlin=nn.ReLU(), context=False, h_dim=None, c_dim=None):
        
        super(vgg_encoder, self).__init__()
        
        kernel_list, pool_list, stride_list = self.check_input(filter_list, kernel_list, pool_list, stride_list, context, c_dim)
        zipped = zip(filter_list, kernel_list, pool_list, stride_list)
        
        
        conv = [vgg_block(i[0], i[1], i[2], i[3], padding, batchnorm, nonlin) for i in zipped]
        self.conv = nn.Sequential(*conv)
        
        self.h_dim = h_dim if h_dim is not None else self.get_flatten_size(filter_list, pool_list)
        self.context = context
        
        self.mu = nn.Sequential(nn.Linear(self.h_dim, z_dim))
        self.sigma = nn.Sequential(nn.Linear(self.h_dim, z_dim), nn.Softplus())
        if self.context:
            self.c = nn.Sequential(nn.Linear(self.h_dim, c_dim))
        
    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, self.h_dim)
        mu = self.mu(h)
        sigma = self.sigma(h)
        c = self.c(h) if self.context else None
        
        return mu, sigma, c
        
    
    def check_input(self, f, k, p, s, co, c):
        assert isinstance(f, list), "filter_list needs to be a list"
        k = self.expand_list(k, len(f)) if not isinstance(k, list) else k
        p = self.expand_list(p, len(f)) if not isinstance(p, list) else p
        s = self.expand_list(s, len(f)) if not isinstance(s, list) else s
        
        assert len(f)==len(k)==len(p)==len(s), "all parameter lists should be of the same size"
        if co:
            assert c is not None, "when using context, context size must be defined"
        return k, p, s
        
    
    def expand_list(self, lst, length):
        return [lst]*length
        
    def get_flatten_size(self, f, p):
        """
        only works if stride=1 and padding='same'
        """
        width, height = 100, 25
        for i in p:
            width = math.floor(width/i)
            height = math.floor(height/i)
        
        return f[-1][1]*width*height 
        
        
class resnet_block(nn.Module):

    def __init__(self, filters, kernel_size, downsample=False ,batchnorm=True, nonlin=nn.ELU(), alpha=0.25):
        
        super(resnet_block, self).__init__()
        
        in_filters, out_filters = filters
        self.alpha = alpha
        self.downsample = downsample
        
        if not downsample:
            assert in_filters == out_filters, "filter size must be the same when not downsampling"
         
        if downsample:
            stride = 2
            self.down = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=stride)
        else:
            stride = 1
        
        block = [nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=1),
                 nn.BatchNorm2d(out_filters) if batchnorm else None,
                 nonlin,
                 nn.Conv2d(out_filters, out_filters, kernel_size=kernel_size, stride=1, padding='same')]
        
        out = [nn.BatchNorm2d(out_filters) if batchnorm else None,
               nonlin]
        
        block = [i for i in block if i is not None]
        out = [i for i in out if i is not None]
        
            
        self.block = nn.Sequential(*block)
        self.out = nn.Sequential(*out)
        
    def forward(self, x):
        if self.downsample:
            identity = self.down(x)
        else:
            identity = x
        h = self.block(x)
        h = identity + self.alpha*h
        return self.out(h)
        
        
class resnet_encoder(nn.Module):
    
    def __init__(self, z_dim, filter_list, kernel_list, downsample_list, stem_filters, stem_kernels, batchnorm=True, 
                  nonlin=nn.ReLU(), context=False, h_dim=None, c_dim=None, alpha=0.25):
        
        super(resnet_encoder, self).__init__()
        kernel_list = self.check_input(filter_list, kernel_list, downsample_list, context, c_dim)    
        
        stem = [nn.Conv2d(stem_filters[0][0], stem_filters[0][1], kernel_size=stem_kernels[0], stride=1, padding='same'),
                nn.BatchNorm2d(stem_filters[0][1]) if batchnorm else None,
                nonlin,
                nn.Conv2d(stem_filters[1][0], stem_filters[1][1], kernel_size=stem_kernels[1], stride=1, padding='same'),
                nn.BatchNorm2d(stem_filters[1][1]) if batchnorm else None,
                nonlin,
                nn.MaxPool2d((2,2))]
        
        stem = [i for i in stem if i is not None]
        self.stem = nn.Sequential(*stem)
        
        zipped = zip(filter_list, kernel_list, downsample_list)
        
        resnet = [resnet_block(i[0],i[1],i[2],batchnorm,nonlin,alpha) for i in zipped]
        self.resnet = nn.Sequential(*resnet)
        
        
        self.h_dim = np.prod(self.resnet(self.stem(torch.zeros(2,5,25,100))).shape[1:])
        #self.h_dim = h_dim if h_dim is not None else self.get_flatten_size(filter_list, downsample_list)
        self.context = context
        
        self.mu = nn.Sequential(nn.Linear(self.h_dim, z_dim))
        self.sigma = nn.Sequential(nn.Linear(self.h_dim, z_dim), nn.Softplus())
        if self.context:
            self.c = nn.Sequential(nn.Linear(self.h_dim, c_dim))
     
    def forward(self, x):
        h = self.stem(x)
        h = self.resnet(h)
        h = h.view(-1, self.h_dim)
        mu = self.mu(h)
        sigma = self.sigma(h)
        c = self.c(h) if self.context else None
        
        return mu, sigma, c
    
    def check_input(self, f, k, d, co, c):
        assert isinstance(f, list), "filter_list needs to be a list"
        assert isinstance(d, list), "downsample_list needs to be a list"
        if not isinstance(k, list):
            k = self.expand_list(k, len(f))
        assert len(f)==len(k)==len(d), "all parameter lists should be of the same size"
        if co:
            assert c is not None, "when using context, context size must be defined"
        
        for i in range(len(d)):
            if not d[i]:
                assert f[i][0] == f[i][1], "in- and out-filters must be the same when not downsampling"
        
        return k
        
    
    def expand_list(self, lst, length):
        return [lst]*length
        
    def get_flatten_size(self, f, d):
        """
        only works if stride=1 and padding='same'
        """
        width, height = 100, 25
        for i in d:
            if d:
                width = math.floor(width/2)
                height = math.floor(height/2)
        
        return f[-1][1]*width*height 
