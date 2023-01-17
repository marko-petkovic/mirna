import torch 
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from decoders import *
from encoders import *
from flow import *
from dataset import *
from predictors import *
from priors import *


class MIRDIVA(nn.Module):
    
    
    def __init__(self, args):
        super(MIRDIVA, self).__init__()

        self.betas = args.betas.copy()
        self.auxs = args.auxs.copy()
        self.rec = args.rec
        self.zs = args.zs
        self.iaf = args.iaf
        
        self.type = 'diva'
        
        self.encs_x = None
        self.encs_y = None
        self.encs_m = None
        self.encs_ym = None
        
        self.flows_x = None
        self.flows_y = None
        self.flows_m = None
        self.flows_ym = None
        
        self.priors_y = None
        self.priors_m = None
        self.priors_ym = None
        
        self.preds_y = None
        self.preds_m = None
        self.preds_y_ym = None
        self.preds_m_ym = None
        
        
    
        self.names = {'x':self.zs[0],'y':self.zs[1],'m':self.zs[2],'ym':self.zs[3]}
        
        self.dec = decoder([i for i in args.zs if i > 0], batchnorm=args.bn,nonlin=args.nonlin,sttng=args.sttng)
        
        
        if self.iaf:
            if self.zs[0] > 0:
                
                self.encs_x =  resnet_encoder(args.zs[0], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3, [False,True,False,True], [(5,32),(32,48)],[5,5],args.bn, args.nonlin, alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
                self.flows_x = Flow(args.zs[0], args.c_dim, args.context, flip=True, flow_blocks=8)
            
            if self.zs[1] > 0:
                
                self.encs_y =  resnet_encoder(args.zs[1], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3, [False,True,False,True], [(5,32),(32,48)],[5,5],args.bn, args.nonlin, alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
                self.flows_y = Flow(args.zs[1], args.c_dim, args.context, flip=True, flow_blocks=8)
            
                self.preds_y = y_cls(args.zs[1])
                self.priors_y = y_prior(args.zs[1], args.bn)
            
            if self.zs[2] > 0:
                
                self.encs_m =  resnet_encoder(args.zs[1], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3, [False,True,False,True], [(5,32),(32,48)],[5,5],args.bn, args.nonlin, alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
                self.flows_m = Flow(args.zs[2], args.c_dim, args.context, flip=True, flow_blocks=8)
                self.preds_m = mfe_cls(args.zs[2])
                self.priors_m = mfe_conv_prior(args.zs[2], args.bn)
                
                
                if self.auxs['y_m'] > 0:
                    self.preds_y_m = y_cls(args.zs[2])
            
            if self.zs[3] > 0:
                
                self.encs_ym = resnet_encoder(args.zs[3], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3, [False,True,False,True], [(5,32),(32,48)],[5,5],args.bn, args.nonlin, alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
                self.flows_ym = Flow(args.zs[3], args.c_dim, args.context, flip=True, flow_blocks=8)
            
                self.preds_y_ym = y_cls(args.zs[3])
                self.preds_m_ym = mfe_cls(args.zs[3])
                self.priors_ym = combined_conv_prior(args.zs[3], args.zs[3], args.bn

                
            

            self.cuda()
            
    def forward(self, x, y, m):
        
        qz, pz, z_0, z_f, ldj, preds = {},{},{},{},{},{}
        
        
        if self.zs[0] > 0:
            i = 'x'
            qz_mu, qz_std, c = self.encs_x(x) 
            qz[i] = dist.Normal(qz_mu, qz_std)
            z_0[i] = qz[i].rsample()
            
            pz_mu, pz_std =  torch.zeros(qz_mu.size()[0], self.zs[0]).cuda(), \
                             torch.ones(qz_mu.size()[0], self.zs[0]).cuda()
            pz[i] = dist.Normal(pz_mu, pz_std)
           
            if self.iaf:
                _z_f, _ldj = self.flows_x(z_0[i], c)
                z_f[i] = _z_f
                ldj[i] = _ldj
            else:
                z_f[i] = z_0[i]
                ldj[i] = 0
            #z_f[i], ldj[i] = self.flows_x(z_0[i], c) if self.iaf else z_0[i], 0

        
        if self.zs[1] > 0:
            i = 'y'
            qz_mu, qz_std, c = self.encs_y(x) 
            qz[i] = dist.Normal(qz_mu, qz_std)
            z_0[i] = qz[i].rsample()
            
            pz_mu, pz_std = self.priors_y(x, y, m)
            pz[i] = dist.Normal(pz_mu, pz_std)
            
            #preds[i] = self.preds_y(z_0[i])
            
            if self.iaf:
                _z_f, _ldj = self.flows_y(z_0[i], c)
                z_f[i] = _z_f
                ldj[i] = _ldj
            else:
                z_f[i] = z_0[i]
                ldj[i] = 0
            
            preds[i] = self.preds_y(z_f[i])
            
        if self.zs[2] > 0:
            i = 'm'
            qz_mu, qz_std, c = self.encs_m(x) 
            qz[i] = dist.Normal(qz_mu, qz_std)
            z_0[i] = qz[i].rsample()
            
            pz_mu, pz_std = self.priors_m(x, y, m)
            pz[i] = dist.Normal(pz_mu, pz_std)
            
            
            
            if self.iaf:
                _z_f, _ldj = self.flows_m(z_0[i], c)
                z_f[i] = _z_f
                ldj[i] = _ldj
            else:
                z_f[i] = z_0[i]
                ldj[i] = 0
            
            preds[i] = self.preds_m(z_f[i])
            
            if self.auxs['y_m'] > 0:
                preds['y_m'] = self.preds_y_m(z_f[i])
        
        if self.zs[3] > 0:
            i = 'ym'
            qz_mu, qz_std, c = self.encs_ym(x) 
            qz[i] = dist.Normal(qz_mu, qz_std)
            z_0[i] = qz[i].rsample()
            
            pz_mu, pz_std = self.priors_ym(x, y, m)
            pz[i] = dist.Normal(pz_mu, pz_std)
            
            preds['y_ym'] = self.preds_y_ym(z_0[i])
            preds['m_ym'] = self.preds_m_ym(z_0[i])
            
            if self.iaf:
                _z_f, _ldj = self.flows_ym(z_0[i], c)
                z_f[i] = _z_f
                ldj[i] = _ldj
            else:
                z_f[i] = z_0[i]
                ldj[i] = 0
            
            preds['y_ym'] = self.preds_y_ym(z_f[i])
            preds['m_ym'] = self.preds_m_ym(z_f[i])
        
            
        # decoding
        x_hat, col, len_bar_top, len_bar_bot = self.dec.forward([z_f[i] for i in z_f])

        return x_hat, preds, qz, pz, z_0, ldj, col, len_bar_top, len_bar_bot
    
    def loss_function(self, x, y, m):
        
        stat_dict = {}
        
        n = x.shape[0]
        
        x_hat, preds, qz, pz, z_0, ldj, _, _, _ = self.forward(x, y, m)
        
        rec_loss = F.mse_loss(x_hat, x, reduction='sum')
        
        stat_dict['rec'] = rec_loss.item()/n
        
        
        total_loss = self.rec*rec_loss
        
        for i in self.names:
            
            if self.names[i] == 0:
                continue
            
            kl_loss = ((qz[i].log_prob(z_0[i]) - pz[i].log_prob(z_0[i])).sum(dim=1) - ldj[i]).sum()
            
            stat_dict[f'kl_loss_{i}'] = kl_loss.item()/n
            stat_dict[f'beta_{i}'] = self.betas[i]
            
            if i == 'y':
                aux = F.cross_entropy(preds['y'], y, reduction='sum')
                #aux_loss['y'] = aux
                acc_y = (preds['y'].argmax(1) == y.argmax(1)).sum().float()
                
                stat_dict[f'aux_loss_y'] = aux.item()/n
                stat_dict[f'acc_y'] = acc_y.item()/n
                
                total_loss += self.auxs[i]*aux
                
            elif i == 'm':
                aux = F.mse_loss(preds['m'], m, reduction='sum')
                #aux_loss['m'] = aux
                
                stat_dict[f'aux_loss_m'] = aux.item()/n
                
                
                total_loss += self.auxs[i]*aux
                
                if self.auxs['y_m'] > 0:
                    
                    aux_y_m = F.cross_entropy(preds['y_m'], y, reduction='sum')
                    acc_y_m = (preds['y_m'].argmax(1) == y.argmax(1)).sum().float()
                    
                    stat_dict[f'aux_loss_y_m'] = aux.item()/n
                    stat_dict[f'acc_y_m'] = acc_y_m.item()/n
                    
                    total_loss += self.auxs['y_m']*aux_y_m
                
            
            elif i == 'ym':
                aux1 = F.cross_entropy(preds['y_ym'], y, reduction='sum')
                aux2 = F.mse_loss(preds['m_ym'], m, reduction='sum')
                #aux_loss['y_ym'] = aux1
                #aux_loss['m_ym'] = aux2
                aux = aux1+aux2
                acc_y_ym = (preds['ym'][0].argmax(1) == y.argmax(1)).sum().float()
                
                stat_dict[f'aux_loss_y_ym'] = aux1.item()/n
                stat_dict[f'acc_y_ym'] = acc_y_ym.item()/n
                stat_dict[f'aux_loss_m_ym'] = aux2.item()/n
                
                total_loss += self.auxs[i][0]*aux1
                total_loss += self.auxs[i][1]*aux2
                
            total_loss += self.betas[i]*kl_loss
        
        stat_dict['loss']=total_loss.item()/n
            
            
        return total_loss, stat_dict
        
            
            
            
        
class MIRVAE(nn.Module):
    
    def __init__(self, args):
        super(MIRVAE, self).__init__()
        
        self.beta = args.beta
        self.rec = args.rec
        self.iaf = args.iaf
        self.z = args.z
        
        self.type = 'vae'
        
        if not self.iaf:
            self.enc = vgg_encoder(args.z, [(5,args.f1),(args.f1, args.f2), (args.f2,args.f3)], [args.k1, args.k2, args.k3], args.pool, args.stride, args.padding, args.bn, args.nonlin)
        
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
        
        
        
        
        
        
#         self.args = args
        
#         self.create_encoders()
        
    
    
#     def create_vgg(self, z_dim)
    
#     def create_vgg_iaf(self, )
    
#     def create_flow(self)
    
#     def create_resnet(self, z_dim):
        
    
#     def create_encoders(self):
        
#         self.lat_space = {}
#         self.encs = {}
        
        
        
        
#         # definition of latent space for zmy
        
#         if self.args.zmy > 0:    
#             self.lat_space['zmy'] = (self.args.zmy, self.args.zmy_loss)
            
#             if self.args.vgg and not self.args.iaf:
#                 enc = vgg_encoder(self.args.zmy, [(5,self.args.f1),(self.args.f1, self.args.f2), (self.args.f2,self.args.f3)], [self.args.k1, self.args.k2, self.args.k3], self.args.pool, self.args.stride, self.args.padding, self.args.bn, self.args.nonlin)
#                 iaf = None
            
                
                
#             elif self.args.vgg and self.args.iaf:
                
#                 enc = vgg_encoder(self.args.zmy, [(5,self.args.f1),(self.args.f1, self.args.f2), (self.args.f2,self.args.f3)], [self.args.k1, self.args.k2, self.args.k3], self.args.pool, self.args.stride, self.args.padding, self.args.bn, self.args.nonlin, context=self.args.context, h_dim=None, c_dim=self.args.c_dim)
                
#                 iaf = Flow(self.args.zmy, self.args.context, self.args.c_dim, flip=True, flow_blocks=8)
                
            
            
#             if self.args.res and not self.args.iaf:
                
#                 enc = resnet_encoder(self.args.zmy, [(self.args.f1,self.args.f1),(self.args.f1, self.args.f2), (self.args.f2,self.args.f2),(self.args.f2,self.args.f3)], 3, [False,True,False,True], [(5,64),(64,64)],[5,5],self.args.bn, self.args.nonlin, alpha=self.args.res_alpha)
                
#                 iaf = None
                
#             elif self.args.res and self.args.iaf:
                
#                 enc = resnet_encoder(self.args.zmy, [(self.args.f1,self.args.f1),(self.args.f1, self.args.f2), (self.args.f2,self.args.f2),(self.args.f2,self.args.f3)], 3, [False,True,False,True], [(5,64),(64,64)],[5,5],self.args.bn, self.args.nonlin, alpha=self.args.res_alpha)
                
#                 iaf = None
                
            
            
            
            
            
            
#             aux_mfe = mfe_cls(self.args.zmy)
#             aux_y = y_cls(self.args.zmy)
                
#             self.encs['zmy'] = [enc, iaf, aux_y, aux_mfe]
        
#         if self.args.zm > 0:
#             self.lat_space['zm'] = (self.args.zm, self.args.zm_loss)
            
            
#         if self.args.zy > 0:
#             self.lat_space['zy'] = (self.args.zx, self.args.zy_loss)
            
#         if self.args.zx > 0:
#             self.lat_space['zx'] = (self.args.zx, self.args.zx_loss)
        
        
#         self.enc = {}
        
#         for i in self.args.z_info:
#             z_info = self.args.z_info[i]
            
            
        