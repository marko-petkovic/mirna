import torch
import torch.nn as nn


class standard_normal_prior(nn.Module):
    
    def __init__(self, z_dim):
        super(standard_normal_prior, self).__init__()
        self.z_dim = z_dim
        
    def forward(self, x):
        mu =  torch.zeros(x.shape[0], z_dim).to('cuda')
        sigma =  torch.ones(x.shape[0], z_dim).to('cuda')
        
        return mu, sigma

class mfe_conv_prior(nn.Module):
    
    def __init__(self, z_dim, batchnorm=True, nonlin=nn.LeakyReLU()):
        
        super(mfe_conv_prior, self).__init__()
        
        conv = [nn.Conv1d(2,8,kernel_size=13),
                nn.BatchNorm1d(8) if batchnorm else None,
                nonlin,
                nn.Conv1d(8,16,kernel_size=9),
                nn.BatchNorm1d(16) if batchnorm else None,
                nonlin,
                nn.Conv1d(16,32,kernel_size=5),
                nn.BatchNorm1d(32) if batchnorm else None,
                nonlin,
                nn.Conv1d(32,64,kernel_size=5),
                nn.BatchNorm1d(64) if batchnorm else None,
                nonlin,
                nn.Conv1d(64,32,kernel_size=5),
                nn.BatchNorm1d(32) if batchnorm else None,
                nonlin,
                nn.Conv1d(32,32,kernel_size=5),
                nn.BatchNorm1d(32) if batchnorm else None,
                nonlin,
                nn.Conv1d(32,1,kernel_size=1),
                nonlin]
        
        conv = [i for i in conv if i is not None]
        self.conv = nn.Sequential(*conv)
        
        self.mu = nn.Sequential(nn.Linear(64, z_dim))
        self.sigma = nn.Sequential(nn.Linear(64, z_dim,), nn.Softplus())
        
        
    def forward(self, x, y, m):
        
        h = self.conv(m)[:,0]
        mu = self.mu(h)
        sigma = self.sigma(h)
        return mu, sigma
    
class y_prior(nn.Module):
    
    def __init__(self, z_dim, batchnorm=True, nonlin=nn.LeakyReLU()):
        
        super(y_prior, self).__init__()
        
        fc = [nn.Linear(2, z_dim),
              nn.BatchNorm1d(z_dim) if batchnorm else None,
              nonlin,
              nn.Linear(z_dim, z_dim),
              nn.BatchNorm1d(z_dim) if batchnorm else None,
              nonlin]
        fc = [i for i in fc if i is not None]
        
        self.fc = nn.Sequential(*fc)
        self.mu = nn.Sequential(nn.Linear(z_dim, z_dim))
        self.sigma = nn.Sequential(nn.Linear(z_dim, z_dim), nn.Softplus())
        
    def forward(self, x, y, m):
        h = self.fc(y)
        mu = self.mu(h)
        sigma = self.sigma(h)
        return mu, sigma