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

class MIRDIVA(nn.Module):
    
    
    def __init__(self, args):
        super(MIRDIVA, self).__init__()

        self.betas = args.betas.copy()
        self.auxs = args.auxs.copy()
        self.rec = args.rec
        self.zs = args.zs
        self.iaf = args.iaf
        
        self.type = 'diva'
        
        self.names = {'x':self.zs[0],'y':self.zs[1],'m':self.zs[2]}
        
        self.dec = decoder([i for i in args.zs if i > 0], batchnorm=args.bn,nonlin=args.nonlin,sttng=args.sttng)
        
        # create encoders with flows
        if self.iaf:
            self.encs_x =  resnet_encoder(args.zs[0], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3,
                                          [False,True,False,True], [5,32,48], [5,5], args.bn, args.nonlin,
                                          alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
            self.flows_x = Flow(args.zs[0], args.c_dim, args.context, flip=True, flow_blocks=8)

                
            self.encs_y =  resnet_encoder(args.zs[1], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3,
                                          [False,True,False,True], [5,32,48], [5,5], args.bn, args.nonlin,
                                          alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
            self.flows_y = Flow(args.zs[1], args.c_dim, args.context, flip=True, flow_blocks=8)


            self.encs_m =  resnet_encoder(args.zs[1], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3,
                                          [False,True,False,True], [5,32,48], [5,5], args.bn, args.nonlin,
                                          alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
            self.flows_m = Flow(args.zs[2], args.c_dim, args.context, flip=True, flow_blocks=8)
        
        # create encoders without flows
        else:
            self.encs_x =  resnet_encoder(args.zs[0], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3,
                                          [False,True,False,True], [5,32,48], [5,5], args.bn, args.nonlin,
                                          alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)

            self.encs_y =  resnet_encoder(args.zs[1], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3,
                                          [False,True,False,True], [5,32,48], [5,5], args.bn, args.nonlin,
                                          alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)

            self.encs_m =  resnet_encoder(args.zs[1], [(args.f1,args.f1),(args.f1, args.f2), (args.f2,args.f2),(args.f2,args.f3)], 3,
                                          [False,True,False,True], [5,32,48], [5,5], args.bn, args.nonlin,
                                          alpha=args.res_alpha, context=args.context, h_dim=None, c_dim=args.c_dim)
        
        
        # create priors and auxiliary predictors
        self.preds_y = y_cls(args.zs[1])
        self.priors_y = y_prior(args.zs[1], args.bn)

        self.preds_m = mfe_cls(args.zs[2])
        self.priors_m = mfe_conv_prior(args.zs[2], args.bn)

        if self.auxs['y_m'] > 0:
            self.preds_y_m = y_cls(args.zs[2])


        self.cuda()
            
    def forward(self, x, y, m):
        
        qz, pz, z_0, z_f, ldj, preds = {},{},{},{},{},{}
        
        # encode x
        qz_mu, qz_std, c_x = self.encs_x(x) 
        qz['x'] = dist.Normal(qz_mu, qz_std)
        z_0['x'] = qz['x'].rsample()
        
        pz_mu, pz_std =  torch.zeros(qz_mu.size()[0], self.zs[0]).cuda(), \
                         torch.ones(qz_mu.size()[0], self.zs[0]).cuda()
        pz['x'] = dist.Normal(pz_mu, pz_std)
        
        # encode y
        qz_mu, qz_std, c_y = self.encs_y(x) 
        qz['y'] = dist.Normal(qz_mu, qz_std)
        z_0['y'] = qz['y'].rsample()
        
        pz_mu, pz_std = self.priors_y(x, y, m)
        pz['y'] = dist.Normal(pz_mu, pz_std)
        
        # encode m
        qz_mu, qz_std, c_m = self.encs_m(x) 
        qz['m'] = dist.Normal(qz_mu, qz_std)
        z_0['m'] = qz['m'].rsample()
        
        pz_mu, pz_std = self.priors_m(x, y, m)
        pz['m'] = dist.Normal(pz_mu, pz_std)
        
        
        # process flows
        if self.iaf:
            # flow x
            _z_f, _ldj = self.flows_x(z_0['x'], c_x)
            z_f['x'] = _z_f
            ldj['x'] = _ldj
            
            # flow y
            _z_f, _ldj = self.flows_x(z_0['y'], c)
            z_f['y'] = _z_f
            ldj['y'] = _ldj
            
            # flow m
            _z_f, _ldj = self.flows_x(z_0[i], c)
            z_f[i] = _z_f
            ldj[i] = _ldj
            
        else:
            z_f['x'] = z_0['x']
            ldj['x'] = 0
            
            z_f['y'] = z_0['y']
            ldj['y'] = 0
            
            z_f['m'] = z_0['m']
            ldj['m'] = 0
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
        