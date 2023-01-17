import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F



class bar_length_conv(nn.Module):
    """
    input dimension must be 1152 when using this decoder
    """
    def __init__(self, batchnorm=True, nonlin=nn.ReLU()):
        super(bar_length_conv, self).__init__()
        
        bar_length = [nn.ConvTranspose2d(32,32,kernel_size=3,stride=3,padding=0),
                      nn.BatchNorm2d(32) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose2d(32,32,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm2d(32) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose2d(32,64,kernel_size=3,stride=3,padding=0),
                      nn.BatchNorm2d(64) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm2d(64) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose2d(64,32,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm2d(32) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose2d(32,1,kernel_size=1,stride=1,padding=0)]
        bar_length = [i for i in bar_length if i is not None]
        
        self.bar_length = nn.Sequential(*bar_length)
        
    def forward(self, h):
        
        #h = h.view(-1,32,12,3)
        return self.bar_length(h)[:,:,4:-4,1:]
    
class bar_color_conv(nn.Module):
    """
    input dimension must be 1152 when using this decoder
    """
    def __init__(self, batchnorm=True, nonlin=nn.ReLU()):
        super(bar_color_conv, self).__init__()
        
        bar_color = [nn.ConvTranspose1d(32,32,kernel_size=3,stride=3,padding=0),
                      nn.BatchNorm1d(32) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose1d(32,32,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm1d(32) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose1d(32,64,kernel_size=3,stride=3,padding=0),
                      nn.BatchNorm1d(64) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose1d(64,64,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm1d(64) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose1d(64,32,kernel_size=3,stride=1,padding=1),
                      nn.BatchNorm1d(32) if batchnorm else None,
                      nonlin,
                      nn.ConvTranspose1d(32,10,kernel_size=1,stride=1,padding=0)]
        bar_color = [i for i in bar_color if i is not None]
        
        self.bar_color = nn.Sequential(*bar_color)
        
    def forward(self, h):
        
        h = h.sum(dim=3)
        col = self.bar_color(h)[:,:,None,4:-4]
        col = torch.cat([col[:,:5,:,:],col[:,5:,:,:]], dim=2)
        return col
    
class bar_length_fc(nn.Module):
    
    def __init__(self, in_dim=512):
        super(bar_length_fc, self).__init__()
        
        self.bar_length = nn.Linear(in_dim, 2600)
    
    def forward(self, h):
        
        h = self.bar_length(h)
        return h.view(-1,1,100,26)
    
    
class bar_color_fc(nn.Module):
    
    def __init__(self, in_dim=512):
        super(bar_color_fc, self).__init__()
        
        self.bar_color = nn.Linear(in_dim, 1000)
    
    def forward(self, h):
        return self.bar_color(h).view(-1,5,2,100)
        
    

class decoder(nn.Module):
    def __init__(self, z_dims=[64,128,64], dim1=3*12*32, dim2=512, batchnorm=True, drop1=0.1, drop2=0.2, nonlin=nn.ReLU(), sttng='conv'):
        super(decoder, self).__init__()

        self.drop1 = drop1
        self.drop2 = drop2
        self.sttng = sttng
        
        
        if self.sttng == 'conv':
            self.fc = nn.Linear(sum(z_dims), dim1)
            self.bn = nn.BatchNorm2d(32) if batchnorm else nn.Sequential()
            self.nonlin = nonlin
            
            self.color = bar_color_conv(batchnorm, nonlin)
            self.length = bar_length_conv(batchnorm, nonlin)
            
        elif self.sttng == 'fc':
            fc = [nn.Linear(sum(z_dims), dim1),
                  nn.BatchNorm1d(dim1) if batchnorm else None, 
                  nonlin,
                  nn.Dropout(drop1) if drop1 > 0 else None,
                  nn.Linear(dim1, dim2),
                  nn.BatchNorm1d(dim2) if batchnorm else None,  
                  nonlin,
                  nn.Dropout(drop2) if drop2 > 0 else None
                 ]
            
            self.fc = nn.Sequential(*fc)
            
            self.color = bar_color_fc(dim2)
            self.length = bar_length_fc(dim2)
        
        self.stamp = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                                   [1,1,0,0,0,0,0,0,0,0,0,0,0],
                                   [1,1,1,0,0,0,0,0,0,0,0,0,0],
                                   [1,1,1,1,0,0,0,0,0,0,0,0,0],
                                   [1,1,1,1,1,0,0,0,0,0,0,0,0],
                                   [1,1,1,1,1,1,0,0,0,0,0,0,0],
                                   [1,1,1,1,1,1,1,0,0,0,0,0,0],
                                   [1,1,1,1,1,1,1,1,0,0,0,0,0],
                                   [1,1,1,1,1,1,1,1,1,0,0,0,0],
                                   [1,1,1,1,1,1,1,1,1,1,0,0,0],
                                   [1,1,1,1,1,1,1,1,1,1,1,0,0],
                                   [1,1,1,1,1,1,1,1,1,1,1,1,0],
                                   [1,1,1,1,1,1,1,1,1,1,1,1,1],
                                 ])[None,:].to('cuda').float()
        
        
    def forward(self, z):
        if len(z) > 1:
            z = torch.cat(z, dim=1)
        else:
            z = z[0]
        
        if self.sttng == 'fc':
            h = self.fc(z)
        elif self.sttng == 'conv':
            h = self.fc(z)
            h = h.view(-1,32,12,3)
            h = self.bn(h)
            h = self.nonlin(h)
        
        
        
        col = nn.Softmax(dim=1)(self.color(h))
        len_bar = self.length(h)                       
        
        # create "distribution" over image for bars
        len_bar_top_ = nn.Softmax(dim=2)(len_bar[:,0,:,:13]).flip(2)
        len_bar_top = torch.bmm(len_bar_top_, self.stamp.repeat(len_bar.shape[0],1,1)).flip(2)[:,None]
        
        len_bar_bot_ = nn.Softmax(dim=2)(len_bar[:,0,:,13:])
        len_bar_bot = torch.bmm(len_bar_bot_, self.stamp.repeat(len_bar.shape[0],1,1))[:,None,:,:12]
        
        len_bar_ = torch.cat([len_bar_top, len_bar_bot], 3).permute(0,1,3,2).repeat(1,5,1,1)
        
        # create "distribution" over image for color
        col_top = col[:,:,0,None,:].repeat(1,1,13,1)
        col_bot = col[:,:,1,None,:].repeat(1,1,12,1)
        color = torch.cat([col_top,col_bot],2)
        
        rna = color*len_bar_
        
        return rna, col, len_bar_top_, len_bar_bot_
    
    def sample(self, color, len_bar_t, len_bar_b, mean=True, cat=False):
        

        if cat:
            color  = color.permute(0,2,3,1)
            mt = torch.argmax(len_bar_t, dim=2)#, keepdim=True)
            bt = torch.argmax(len_bar_b, dim=2)#, keepdim=True)
            co = torch.argmax(color, dim=3)#, keepdim=True)
            
            len_bar_top_ = F.one_hot(mt, num_classes=13).float()
            len_bar_bot_ = F.one_hot(bt, num_classes=13).float()
            col = F.one_hot(co, num_classes=5).permute(0,3,1,2).float()            
            
            col_top = col[:,:,0,None,:].repeat(1,1,13,1)
            col_bot = col[:,:,1,None,:].repeat(1,1,12,1)
            colors = torch.cat([col_top,col_bot],2)
            
            
            len_bar_top = torch.bmm(len_bar_top_, 
                                    self.stamp.repeat(len_bar_top_.shape[0],1,1)).flip(2)[:,None]
            len_bar_bot = torch.bmm(len_bar_bot_,
                                    self.stamp.repeat(len_bar_bot_.shape[0],1,1))[:,None,:,:12]
            len_bar_ = torch.cat([len_bar_top, len_bar_bot], 3 \
                                ).permute(0,1,3,2).repeat(1,5,1,1)
        
            return colors*len_bar_
        
        if mean:
            bars_t = torch.argmax(len_bar_t, dim=2)
            bars_b = torch.argmax(len_bar_b, dim=2)
            col = torch.argmax(color, dim=1)
        
        else:
            bars_t = dist.Categorical(len_bar_t).sample()
            bars_b = dist.Categorical(len_bar_b).sample()
            col = dist.Categorical(color).sample()
            
            
        
        # since we do not have bars with length 1, increase length by 1 for "existing" bars
        # this is in conformance with how the "stamp" works
        bars_t = torch.where(bars_t>0, bars_t+1, bars_t)
        bars_b = torch.where(bars_b>0, bars_b+1, bars_b)
        
        if cat:
            out = torch.zeros((color.shape[0], 25, 100, 5))
            for i in range(color.shape[0]):
                for j in range(100):
                    out[i, 13-bars_t[i,j]:13, j] = self.get_color_cat(col[i,0,j])
                    out[i, 13:13+bars_b[i,j], j] = self.get_color_cat(col[i,1,j])
            
            out = out.permute(0,3,1,2)
        
        else:
            out = torch.ones((color.shape[0], 25, 100 ,3))
            for i in range(color.shape[0]):
                for j in range(100):
                    out[i, 13-bars_t[i,j]:13, j] = self.get_color(col[i,0,j])
                    out[i, 13:13+bars_b[i,j], j] = self.get_color(col[i,1,j])

        return out
                
                

                
    def get_color(self, color):
        if color == 0:
            return torch.tensor([0,0,0])
        elif color == 1:
            return torch.tensor([1,0,0])
        elif color == 2:
            return torch.tensor([0,0,1])
        elif color == 3:
            return torch.tensor([0,1,0])
        elif color == 4:
            return torch.tensor([1,1,0])
        
    def get_color_cat(self, color):
        if color == 0:
            return torch.tensor([1,0,0,0,0])
        elif color == 1:
            return torch.tensor([0,1,0,0,0])
        elif color == 2:
            return torch.tensor([0,0,1,0,0])
        elif color == 3:
            return torch.tensor([0,0,0,1,0])
        elif color == 4:
            return torch.tensor([0,0,0,0,1])
        