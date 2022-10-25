import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

def process_args(args):
    '''
    changes structure of args dictionary
    '''
    nonlin_dict = {'relu':nn.ReLU(),'leakyrelu':nn.LeakyReLU(),'elu':nn.ELU()}
        # modifications to deliver everything in the correct way
    args.zs = [args.zx_dim, args.zy_dim,args.zm_dim, args.zym_dim]
    args.auxs = {'y':args.aux_y, 'm':args.aux_m, 'y_m':args.aux_y_m,
                 'ym':[args.aux_y_ym, args.aux_m_ym]}
    args.betas = {'x':args.beta_x, 'y':args.beta_y,
                 'm':args.beta_m, 'ym':args.beta_ym}
    args.nonlin = nonlin_dict[args.nonlin]
    return args



def save_reconstructions(model_folder, epoch, data_loader, model, estr=''):
    '''
    generates reconstructions during model training
    '''
    batch = next(enumerate(data_loader))
    with torch.no_grad():
        model.eval()
        x = batch[1][0][:10].to('cuda').float()
        y = batch[1][2][:10].to('cuda').float()
        m = batch[1][3][:10].to('cuda').float()
        x_org = batch[1][1][:10]
        
        if model.type == 'vae':
            _, _, _, _, _, color, bar_t, bar_b = model(x,y,m)
        elif model.type == 'diva':
            _, _, _, _, _, _, color, bar_t, bar_b = model(x,y,m)
        
        
        #x_hat, qz, pz, z_0, ldj, col, len_bar_top, len_bar_bot
        
        rec = model.dec.sample(color, bar_t, bar_b)
        
    plt.figure(figsize=(80,20))
    fig, ax = plt.subplots(nrows=10, ncols=2)

    ax[0,0].set_title("Original")
    ax[0,1].set_title("Reconstructed")

    for i in range(rec.shape[0]):
        ax[i, 1].imshow(rec[i].cpu())
        ax[i, 0].imshow(x_org[i].cpu().permute(1,2,0))
        ax[i, 0].xaxis.set_visible(False)
        ax[i, 0].yaxis.set_visible(False)
        ax[i, 1].xaxis.set_visible(False)
        ax[i, 1].yaxis.set_visible(False)
    fig.tight_layout(pad=0.1)
    plt.savefig(f'{model_folder}/reconstructions/e{epoch}{estr}.png')
    plt.close('all')
    

def calculate_error_statistics(x, x_hat):
    '''
    function which calculates all reconstruction statistics
    '''
    correct_nt = correct_nucleotides(x, x_hat)
    correct_sh = correct_shape(x, x_hat)
    correct_le = correct_length(x, x_hat)
    
def correct_nucleotides(x, x_hat):
    '''
    calculates which nucleotides are correct
    '''
    return x[:,:,12:13]==x_hat[:,:,12:13]

def correct_shape(x, x_hat):
    '''
    calculates how correct the shape is
    '''
    x, x_hat = x.sum(dim=(1,2)), x_hat.sum(dim=(1,2))
    return x == x_hat

def correct_length(x, x_hat):
    '''
    calculates which part of the length of the reconstruction is correct
    '''
    x, x_hat = x[:,:,12:13].sum(dim=1), x_hat[:,:,12:13].sum(dim=1) 
    return x == x_hat
    
def set_betas(model, args, epoch):
    '''
    modifiess beta based on (pre)warmup and current epoch
    '''
    if model.type == 'diva':
        for i in model.betas:
            model.betas[i] = set_beta(args.betas[i], model.betas[i], args, epoch)
    elif model.type == 'vae':
        model.beta = set_beta(args.beta, model.beta, args, epoch)
        
    
def set_beta(target_beta, beta, args, epoch):
    '''
    modifies a single beta
    '''
    if epoch < args.prewarmup:
        new_beta = target_beta/args.warmup
    else:
        new_beta = min(target_beta, target_beta * (epoch - args.prewarmup * 1.) / (args.warmup))
    return new_beta
        
    
def model_analysis(model, args, data):
    '''
    gives latent space and reconstruction for a model
    '''
    if model.type == 'vae':
        z, x_hat, x = vae_analysis(model, args, data)
    elif model.type == 'diva':
        z, x_hat, x = diva_analysis(model, args, data)
    
    return z, x_hat, x
        
        
def vae_analysis(model, args, data):
    model.eval()
    z_mu = np.zeros((len(data.dataset), model.z))
    x_hat = np.zeros((len(data.dataset),5,25,100))
    x_rna = np.zeros((len(data.dataset),5,25,100))
    pbar = tqdm(enumerate(data), unit="batch") 
                                     
    for batch_idx, (x, _, y, m) in pbar:
        
        b = x.shape[0]
        # To device
        x, y, m = x.to('cuda').float(), y.to('cuda').float(), m.to('cuda').float()
        
        with torch.no_grad():
            m, s, c = model.enc(x)
            if model.iaf:
                m, _ = model.flow(m, c)

            _, c, lt, lb = model.dec([m])

        rec = model.dec.sample(c, lt, lb, cat=True)
        
        idx = data.batch_size*batch_idx 
        z_mu[idx:idx+b] = m.cpu().numpy()
        x_hat[idx:idx+b] = rec.cpu().numpy()
        x_rna[idx:idx+b] = x.cpu().numpy()
    
    z = {'x':z_mu}
    
    return z, x_hat, x_rna

def diva_analysis(model, args, data):
    model.eval()
    
    #z_mu = torch.zeros((len(data.dataset), model.z)).to('cuda')
    x_hat = np.zeros((len(data.dataset),5,25,100))
    x_rna = np.zeros((len(data.dataset),5,25,100))
    
    n = len(data.dataset)
    if args.zs[0] > 0:
        
        zx = np.zeros((n, args.zs[0]))
            
    if args.zs[1] > 0:
        
        zy = np.zeros((n, args.zs[1]))
                
    if args.zs[2] > 0:
        
        zm = np.zeros((n, args.zs[2]))
            
    if args.zs[3] > 0:
        
        zym = np.zeros((n, args.zs[3]))
            
    pbar = tqdm(enumerate(data), unit="batch") 
                                     
    for batch_idx, (x, _, y, m) in pbar:
        
        b = x.shape[0]
        # To device
        x, y, m = x.to('cuda').float(), y.to('cuda').float(), m.to('cuda').float()
        idx = data.batch_size*batch_idx 

        with torch.no_grad():
            
            _z = []
            if args.zs[0] > 0:
                mx = diva_encoder(x, model.iaf,
                                  model.encs_x.to('cuda'),
                                  model.flows_x.to('cuda'))
                _z.append(mx)
                zx[idx:idx+b] = mx.cpu().numpy()
            
            
            if args.zs[1] > 0:
                my = diva_encoder(x, model.iaf,
                                  model.encs_y.to('cuda'),
                                  model.flows_y.to('cuda'))
                _z.append(my)
                zy[idx:idx+b] = my.cpu().numpy()
                
            
            if args.zs[2] > 0:
                mm = diva_encoder(x, model.iaf,
                                  model.encs_m.to('cuda'),
                                  model.flows_m.to('cuda'))
                _z.append(mm)
                zm[idx:idx+b] = mm.cpu().numpy()
                
                
            if args.zs[3] > 0:
                mym = diva_encoder(x, model.iaf,
                                   model.encs_ym.to('cuda'),
                                   model.flows_ym.to('cuda'))
                _z.append(mym)
                zym[idx:idx+b] = mym.cpu().numpy()
            

            _, c, lt, lb = model.dec(_z)

        rec = model.dec.sample(c, lt, lb, cat=True) 
        x_hat[idx:idx+b] = rec.cpu().numpy()
        x_rna[idx:idx+b] = x.cpu().numpy()
    
    z = {}
    
    if args.zs[0] > 0:
        z['x'] = zx
    if args.zs[1] > 0:
        z['y'] = zy
    if args.zs[2] > 0:
        z['m'] = zm
    if args.zs[3] > 0:
        z['ym'] = zym
    
    return z, x_hat, x_rna

def diva_encoder(x, iaf, enc, flow):
    
    m, s, c = enc(x)
    if iaf:
        m, _ = flow(m,c)
    return m