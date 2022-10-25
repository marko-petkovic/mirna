import torch
import torch.nn as nn


class mfe_cls(nn.Module):
    def __init__(self, zm_dim):
        super(mfe_cls, self).__init__()

        self.fc = nn.Sequential(nn.Linear(zm_dim, 64),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(),
                                 )
        self.cls = nn.Sequential(nn.ConvTranspose1d(1,16, kernel_size=5),
                                 nn.BatchNorm1d(16),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose1d(16,32, kernel_size=5),
                                 nn.BatchNorm1d(32),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose1d(32,64, kernel_size=5),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose1d(64,64, kernel_size=5),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose1d(64,32, kernel_size=5),
                                 nn.BatchNorm1d(32),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose1d(32,16, kernel_size=7),
                                 nn.BatchNorm1d(16),
                                 nn.LeakyReLU(),
                                 nn.ConvTranspose1d(16,2, kernel_size=11),
                                 nn.Sigmoid())


        
    def forward(self, zm):

        h = self.fc(zm)
        m_hat = self.cls(h[:,None])  

        return m_hat

class y_cls(nn.Module):
    def __init__(self, zy_dim, simple=True):
        super(y_cls, self).__init__()

        
        if simple:
            self.cls = nn.Sequential(nn.Linear(zy_dim, 2),
                                     nn.Softmax(1))

        else:
            self.cls = nn.Sequential(nn.Linear(zy_dim, zy_dim),
                                 nn.BatchNorm1d(zy_dim),
                                 nn.Sigmoid(),
                                 nn.Dropout(.5),
                                 nn.Linear(zy_dim, 2),
                                 nn.Softmax(1))

        
    def forward(self, zy):
        
        y_hat = self.cls(zy)
        
        return y_hat