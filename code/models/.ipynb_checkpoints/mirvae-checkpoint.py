import torch 
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from components.decoders import *
from components.encoders import *
from components.flow import *
from components.predictors import *
from components.priors import *

from dataset import *


class MIRVAE(nn.Module):
    
    def __init__(self, args):
        super(MIRVAE, self).__init__()
        
        self.beta = args.beta
        self.rec = args.rec
        self.iaf = args.iaf
        self.z = args.z
        
        self.type = 'vae'
        
        if not self.iaf:
            self.enc = resnet_encoder(args.z, [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3, [False,True,False,True], [(5,32),(32,48)],[5,5],args.bn, args.nonlin, alpha=args.res_alpha, context=False, h_dim=None)
        else:
            self.enc = resnet_encoder(args.z, [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3, [False,True,False,True], [(5,32),(32,48)],[5,5],args.bn, args.nonlin, alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
            self.flow = Flow(args.z, args.c_dim, args.context, flip=True, flow_blocks=8)
            
        
        self.dec = decoder([args.z], batchnorm=args.bn,nonlin=args.nonlin, sttng=args.sttng)
        
        self.cuda()
        
        
    def forward(self, x, y, m):

        z_mu, z_std, c = self.enc(x)
        qz = dist.Normal(z_mu, z_std)
        z_0 = qz.rsample()
        
        if self.iaf:
            z_f, ldj = self.flow(z_0, c) 
        else:
            z_f = z_0
            ldj = 0
        
        x_hat, col, len_bar_top, len_bar_bot = self.dec([z_f])
        
        pz_mu, pz_std = torch.zeros(z_f.size()[0], self.z).cuda(),\
                        torch.ones(z_f.size()[0], self.z).cuda()
        pz = dist.Normal(pz_mu, pz_std)
        
        return x_hat, qz, pz, z_0, ldj, col, len_bar_top, len_bar_bot
    
    
    def loss_function(self, x, y, m):
        
        x_hat, qz, pz, z_0, ldj, _, _, _ = self.forward(x, y, m)
        
        rec_loss = F.mse_loss(x_hat, x, reduction='sum')
        
        kl_loss = ((qz.log_prob(z_0) - pz.log_prob(z_0)).sum(dim=1) - ldj).sum()
        
        total_loss = self.rec * rec_loss + self.beta * kl_loss
        
        n = x.shape[0] 
        
        stat_dict = {'loss':total_loss.item()/n,'rec':rec_loss.item()/n, 'kl': kl_loss.item()/n, 'beta':self.beta}
        
        return total_loss, stat_dict
        