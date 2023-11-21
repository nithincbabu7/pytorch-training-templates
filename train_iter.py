import argparse
import os
from os.path import join
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

import random
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

from datasets.dataset_classes import CustomDataset
from graphs.losses.custom_losses import CustomLoss


def get_args():
    parser = argparse.ArgumentParser('Title')

    parser.add_argument('--device', default='cuda', type=str, help='Device to load the models and data in. (cuda/cuda:0/cuda:1,3/cpu)')
    parser.add_argument('--random_seed', default=0, type=int, help='Randomisation seed number for reproducing results')

    # Models will be saved in /<results_folder>/<experiment_name>/
    # Tensorboard plots will be saved in /<results_folder>/runs/<experiment_name>/
    parser.add_argument('--results_folder', default='./results/', type=str, help='Save the results in the following directory')
    parser.add_argument('--exp_name', default='test_iter_ca', type=str, help='Save the results in the following directory')

    parser.add_argument('--loss_names', default=['loss_fn_1', 'loss_fn_2', 'loss_fn_3'], type=str, nargs='+', 
                        help='Loss function names to be used in Tensorboard visualization')
    parser.add_argument('--lambda_list', default=[1.0, 0.1, 0.0], type=float, nargs='+', help='Scaling factors for each loss function')
    parser.add_argument('--warm_start_iter_list', default=[0, 0, 0], type=int, nargs='+', help='Starting iteration for each loss')

    parser.add_argument('--lr', default=1e-2, type=float, help='Optimizer learning rate')
    parser.add_argument('--optimizer', default='adamw', type=str, help='Optimizer (sgd/adam/adamw)')
    parser.add_argument('--weight_decay', default=0, type=float, help='Optimizer weight decay')
    parser.add_argument('--lr_scheduler', default='none', type=str, 
                        help='Optimizer learning rate schedule (none/cosine_annealing/cosine_annealing_warm_restart)')
    parser.add_argument('--cawr_restart_iter', default=200, type=int, help='Restart at cosine annealig at the following itertion')
    parser.add_argument('--lwca_warmup_iter', default=1000, type=int, help='Warmup iterations for linear warmup cosine annealing')

    parser.add_argument('--num_iterations', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size')

    parser.add_argument('--validate', default='none', type=str, help='Validate the model (none/iterations)')
    parser.add_argument('--iter_val_freq', default=100, type=int, help='Validating frequency per iteration')

    parser.add_argument('--iter_save_freq', default=100, type=int, help='Saving frequency per iteration (Do not save if 0)')
    parser.add_argument('--max_iter', default=0, type=int, help='Number of epochs')

    return parser.parse_args()


def main(args):
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    writer = SummaryWriter(log_dir=join(args.results_folder, 'runs', args.exp_name))
    model_loc = join(args.results_folder, args.exp_name)
    os.makedirs(model_loc, exist_ok=True)
    with open(os.path.join(model_loc, 'config_training.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # Initialize all loss terms (only if there is a non zero scaling factor)
    loss_terms = {loss_term: {'loss': torch.empty((1,), device=args.device), 
                              'lambda': args.lambda_list[i],
                              'warm_start_iter': args.warm_start_iter_list[i]} 
                              for i, loss_term in enumerate(args.loss_names) if args.lambda_list[i]}
    # For the first loss, 
    # - access the loss term as `loss_terms['loss_fn_1']['loss']` 
    # - scaling factor as `loss_terms['loss_fn_1']['lambda']`
    
    img_dir = './images/'
    data_csv = './dataset_files/data.csv'
    df_data = pd.read_csv(data_csv)
    df_data = df_data[df_data['set']=='train'].reset_index(drop=True)
    df_data = df_data.astype({'gt': np.float32})    # CSVs stores numbers in float64, but the model is float32
    train_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    train_data = CustomDataset(img_dir, df_data, train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_iter_loader = iter(train_loader)

    if args.validate != 'none':
        df_data_val = pd.read_csv(data_csv)
        df_data_val = df_data_val[df_data_val['set']=='validation'].reset_index(drop=True)
        df_data_val = df_data_val.astype({'gt': np.float32})
        val_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        val_data = CustomDataset(img_dir, df_data_val, val_transform)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        best_perf = -np.inf

    r50 = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(r50, torch.nn.Linear(in_features=1000, out_features=1))
    model = model.to(args.device)
    model.train()

    param_list = [{'params': model.parameters()}]   # append more parameters if needed
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iterations)
    if args.lr_scheduler == 'cosine_annealing_warm_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.cawr_restart_iter)
    if args.lr_scheduler == 'linear_warmup_cosine_annealing':
        lr_lambda = (
            lambda cur_iter: cur_iter / args.lwca_warmup_iter
            if cur_iter <= args.lwca_warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - args.lwca_warmup_iter) / (args.num_iterations - args.lwca_warmup_iter)))
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    if 'loss_fn_1' in loss_terms:
        criterion_1 = CustomLoss()

    for iterations in tqdm(range(1, args.num_iterations+1)):
        try:
            x, y = next(train_iter_loader)
        except StopIteration:
            train_iter_loader = iter(train_loader)
            x, y = next(train_iter_loader)
        
        x, y = x.to(args.device), y.to(args.device)
        optimizer.zero_grad()
        
        loss_name = 'loss_fn_1'
        if loss_name in loss_terms and loss_terms[loss_name]['warm_start_iter'] < iterations:
            y_pred = model(x)
            loss_terms[loss_name]['loss'] = loss_terms[loss_name]['lambda'] * criterion_1(y_pred.view(-1), y)
            writer.add_scalar('loss/%s'%loss_name, loss_terms[loss_name]['loss'].item(), iterations)
        
        loss_name = 'loss_fn_2'
        if loss_name in loss_terms and loss_terms[loss_name]['warm_start_iter'] < iterations:
            y_pred = model(x)
            loss_terms[loss_name]['loss'] = loss_terms[loss_name]['lambda'] * F.mse_loss(y_pred.view(-1), y)
            writer.add_scalar('loss/%s'%loss_name, loss_terms[loss_name]['loss'].item(), iterations)
        
        # Plots learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break
        writer.add_scalar('extra_info/LR', lr, iterations)

        # Sum up all the non-zero losses
        loss = torch.sum(torch.cat([loss_terms[loss_name]['loss'].unsqueeze(0) for loss_name in args.loss_names if loss_name in loss_terms], dim=0))

        loss.backward()
        optimizer.step()
        if args.lr_scheduler != 'none':
            scheduler.step()
        writer.add_scalar('loss', loss.item(), iterations)

        if args.iter_save_freq:
            if iterations % args.iter_save_freq == 0:
                torch.save(model.state_dict(), os.path.join(model_loc, 'model_iter_%06d.pth'%(iterations)))
            
        if args.validate == 'iterations' and iterations % args.iter_val_freq == 0:
            model.eval()
            perf = test_model(model, val_loader, device=args.device)
            writer.add_scalar('val/perf', perf, iterations)
            if perf > best_perf:    # if error metric, multiply it with -1 to get the best model when perf is maximum.
                torch.save(model.state_dict(), os.path.join(model_loc, 'best_iter_model.pth'))
                best_perf = perf
                print('Best model obtained at iteration: %d'%iterations)
            model.train()
        
        # Stop training if number of iterations go beyond max_iter
        if args.max_iter:
            if iterations >= args.max_iter:
                break
    
    writer.close()


def test_model(model, loader, device):
    with torch.no_grad():
        all_y = torch.empty((0, 1), device=device)
        all_y_pred = torch.empty((0, 1), device=device)
        for batch, (x, y) in enumerate(tqdm(loader)):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            all_y = torch.cat((all_y, y.unsqueeze(-1)), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred), dim=0)
        
        score = -F.mse_loss(all_y_pred, all_y)
    
    return score.item()
        



if __name__=='__main__':
    args = get_args()
    main(args)