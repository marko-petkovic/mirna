from models import mirdiva_no_zy, mirdiva, mirvae
from utils import *

import wandb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


from tqdm import tqdm
from tqdm import trange

import datetime
import math
import os
import sys
import numpy as np
import argparse

sys.path.insert(0, 'D:/users/marko/downloads/mirna/models/')


def run_single_epoch(data_loader, model, optimizer, epoch, train=True, n_train_batches=0):

    # to save global epoch statistics
    epoch_stat_dict = {}
    epoch_loss = 0
    
    if train:
        model.train()
        mode = 'train'
    else:
        model.eval()
        mode = 'test'

    pbar = tqdm(enumerate(data_loader), unit="batch", 
                                     desc=f'Epoch {epoch}')
    for batch_idx, (x, _, y, m) in pbar:
        
        
        # To device
        x, y, m = x.to('cuda').float(), y.to('cuda').float(), m.to('cuda').float()
        
        if train:    
            optimizer.zero_grad()
            loss, stat_dict = model.loss_function(x, y, m)
          
            loss.backward()
            optimizer.step()
        
        else:
            with torch.no_grad():
                loss, stat_dict = model.loss_function(x, y, m)
       
        # to save batch statistics
        stat_dict_final = {}
        
        for i in stat_dict:
            # for the first batch, create entries
            if batch_idx==0:
                epoch_stat_dict[i] = [stat_dict[i]]
            else:
                epoch_stat_dict[i].append(stat_dict[i])
            # we want to indicate if its train or test
            stat_dict_final[f'{i}_{mode}'] = stat_dict[i]
        
        pbar.set_postfix(loss=loss.item()/x.shape[0])
        
        if train:
            stat_dict_final[f'batch'] = (epoch-1)*len(data_loader)+batch_idx
        else:
            # to keep n_batches the same
            _batch = (epoch-1)*n_train_batches+batch_idx*n_train_batches/len(data_loader)
            stat_dict_final[f'batch'] = _batch
        
        wandb.log(stat_dict_final)
        
    # calculate mean of all statistics per epoch
    epoch_stat_dict_new = {}
    for i in epoch_stat_dict:
        epoch_stat_dict_new[f'{i}_{mode}_overall'] = np.mean(epoch_stat_dict[i])
    
    epoch_stat_dict_new['epoch'] = epoch
    
    wandb.log(epoch_stat_dict_new)
    
    return epoch_stat_dict_new[f'loss_{mode}_overall']
                        

def train(args, train_loader, test_loader, model, optimizer, end_epoch, start_epoch=0, save_folder='diva',save_interval=5, net_save_interval=50, writer=None):
    
    
    if not os.path.exists(f"{save_folder}/checkpoints/"):
        os.makedirs(f"{save_folder}/checkpoints/")
    
    if not os.path.exists(f"{save_folder}/reconstructions/"):
        os.makedirs(f"{save_folder}/reconstructions/")
    
    #wandb.watch(model)
    train_loss = []
    test_loss = []
    
    
    for epoch in range(start_epoch+1, end_epoch+1):
        
        set_betas(model, args, epoch)
        
        
        loss = run_single_epoch(train_loader, model, optimizer, epoch, train=True)
        train_loss.append(loss)
        
        str_print = "epoch {}: avg train loss {:.2f}".format(epoch, loss)
        print(str_print)

        loss = run_single_epoch(test_loader, model, optimizer, epoch,
                                train=False, n_train_batches=len(train_loader))
        test_loss.append(loss)
       
        str_print = "epoch {}: avg test  loss {:.2f}".format(epoch, loss)
        print(str_print)
        
           
        if epoch % save_interval == 0:
            save_reconstructions(save_folder, epoch, test_loader, model)
            save_reconstructions(save_folder, epoch, train_loader, model, estr='tr')
        
        if epoch % net_save_interval == 0:
            torch.save(model.state_dict(), f'{save_folder}/checkpoints/{epoch}.pth')

    if writer is not None:
        writer.flush()

    torch.save(model.state_dict(), f'{save_folder}/checkpoints/{epoch}.pth')
    
    return train_loss, test_loss

                        
                        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outpath', required=True, type=str, default='./',
                        help='where to save')
    
    parser.add_argument('--projectpath', type=str, default='D:/users/marko/downloads/mirna/')
    parser.add_argument('--generate_data',type=bool, default=False,
                        help='Should the data be generated?')
    parser.add_argument('--model_type', required=True, choices=['vae','diva'])
    
    
    # iaf
    parser.add_argument('--iaf', type=bool, default=False,
                        help='should inverse autoregressive flow be used on the latent space')
    
    parser.add_argument('--context', type=int, default=32,
                        help='amount of nodes used for estimating the context for iaf')
    
    parser.add_argument('--c_dim', type=int, default=1080,
                        help='size of fully connected layer of flow')
    
    # latent dimensions and betas for diva
    parser.add_argument('--zx_dim', type=int, default=64,
                        help='size of latent space for remaining variance')
    parser.add_argument('--zy_dim', type=int, default=64,
                        help='size of latent space for class')
    parser.add_argument('--zm_dim', type=int, default=64,
                        help='size of latent space for mfe')
    parser.add_argument('--zym_dim', type=int, default=0,
                        help='size of latent space for class and mfe')
    
    
    parser.add_argument('--beta_x', type=float, default=0.1)
    parser.add_argument('--beta_y', type=float, default=0.1)
    parser.add_argument('--beta_m', type=float, default=0.1)
    parser.add_argument('--beta_ym', type=float, default=0.1)
    
    parser.add_argument('--aux_y', type=float, default=10)
    parser.add_argument('--aux_m', type=float, default=5)
    parser.add_argument('--aux_y_m', type=float, default=3)
    parser.add_argument('--aux_y_ym', type=float, default=5)
    parser.add_argument('--aux_m_ym', type=float, default=1)
    
    
    # latent dimension for VAE
    parser.add_argument('--z', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.1)
    
    
    # general arguments
    parser.add_argument('--rec', type=float, default=1.0,
                        help='reconstruction error multiplier')
    parser.add_argument('--sttng', choices=['conv', 'fc'], default='conv',
                        help='decoder type')
    
    
    
    # network architecture
    parser.add_argument('--f1', type=int, default=64)
    parser.add_argument('--f2', type=int, default=96)
    parser.add_argument('--f3', type=int, default=128)
    
    parser.add_argument('--k1', type=int, default=7)
    parser.add_argument('--k2', type=int, default=3)
    parser.add_argument('--k3', type=int, default=3)
    
    parser.add_argument('--pool', type=int, default=2, 
                        help='max pooling size')

    parser.add_argument('--stride', type=int, default=1)
    
    parser.add_argument('--padding', type=str, default='same')
    
    parser.add_argument('--bn', type=bool, default=True,
                        help='should batch normalization be used')
    
    parser.add_argument('--nonlin', choices=['relu','leakyrelu','elu'], default='relu')
    parser.add_argument('--res_alpha', type=float, default=0.2,
                        help='multipying factor for non identity part of skip connection')
    
    # training settings
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    
    parser.add_argument('--warmup', type=int, default=250,
                        help='amount of epochs during which beta is being increased to target size')
    parser.add_argument('--prewarmup', type=int, default=0,
                        help='amount of epochs before beta starts being increased')
    
    parser.add_argument('--network_saving_interval', type=int, default=50,
                        help='interval at which weights are saved')
    parser.add_argument('--reconstruction_interval', type=int, default=5,
                        help='interval at which reconstructions are sampled')
    
    # wandb
    parser.add_argument('--model_name', type=str, default='modelo')
    parser.add_argument('--project', type=str, default='MIRGEN')
    parser.add_argument('--entity', type=str, default='generativemirna')
    
    
    
    
    
    
    args = parser.parse_args()
    
    print("parsed arguments")
    
    sys.path.insert(0, args.projectpath)
    
    # wandb login and config generation
    wandb.login(key='46d1be10d4e9900dd55fb752c4ecaa4ca0341b20')
    config = args
    run = wandb.init(name=args.model_name, project=args.project,
                     entity=args.entity, config=config,
                     settings=wandb.Settings(_disable_stats=True,
                                            _disable_meta=True)
                    )
    
    print("initiated run")
    
    args = process_args(args)
    
    print("modified args")
    
    # creating model
    if args.model_type == 'vae':
        model = MIRVAE(args).to('cuda')
    elif args.model_type == 'diva':
        model = MIRDIVA(args).to('cuda')
    
    print("loaded models")
    
    # show a summary of the model to verify everything is correct
    # use two units for bn
    summary(model, [(2,5,25,100), (2,2),(2,2,100)])
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = get_data_loader(args.projectpath, 'train', args.generate_data)
    print("loaded train set")
    test_loader = get_data_loader(args.projectpath, 'test', args.generate_data)
    print("loaded test set")
    train_loss, test_loss = train(args, train_loader, test_loader,
                                  model, optimizer, args.end_epoch, 
                                  start_epoch=args.start_epoch,
                                  save_folder=args.outpath,
                                  save_interval=args.reconstruction_interval, 
                                  net_save_interval=args.network_saving_interval)