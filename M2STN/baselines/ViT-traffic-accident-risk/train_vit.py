import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json
import configparser
import pickle as pkl
from time import time
from datetime import datetime
import shutil
import argparse
import random
import math
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time,get_mask
from lib.early_stop import EarlyStopping
from model.vit import ViT
from lib.utils import mask_loss, compute_loss_vit, predict_and_evaluate_vit

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str,help="test program")
parser.add_argument("--test", action="store_true", help="test program")

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']


all_data_filename = config['all_data_filename']
mask_filename = config['mask_filename']




patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)


train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior

training_epoch = config['training_epoch']

def training(net,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type='nyc'
            ):
    global_step = 1
    for epoch in range(1,training_epoch+1):
        net.train()
        batch = 1
        for train_feature,target_time,gragh_feature,train_label in train_loader:
            start_time = time()
            train_feature,target_time,gragh_feature,train_label = train_feature.to(device),target_time.to(device),gragh_feature.to(device),train_label.to(device)
            l = mask_loss(net(train_feature[:,:,0,:,:], train_feature[:,:,1:,0,0].flatten(start_dim=1,)),train_label,risk_mask,data_type=data_type)#l的shape：(1,)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            training_loss = l.cpu().item()
            


            print('global step: %s, epoch: %s, batch: %s, training loss: %.6f, time: %.2fs'
                % (global_step,epoch, batch, training_loss, time() - start_time),flush=True)
            
            batch+=1
            global_step+=1

        #compute va/test loss
        val_loss = compute_loss_vit(net,val_loader,risk_mask,global_step-1,device,data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' %(global_step-1,epoch,val_loss),flush=True)

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse,test_recall,test_map,test_inverse_trans_pre,test_inverse_trans_label = \
                        predict_and_evaluate_vit(net,test_loader,risk_mask,global_step-1,scaler,device)
           
            high_test_rmse,high_test_recall,high_test_map,_,_ = \
                        predict_and_evaluate_vit(net,high_test_loader,risk_mask,global_step-1,scaler,device)

            print('global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,hihg test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                % (global_step-1,epoch, test_rmse,test_recall,test_map,high_test_rmse,high_test_recall,high_test_map),flush=True)
        
        #early stop according to val loss
        early_stop(val_loss,test_rmse,test_recall,test_map,
                    high_test_rmse,high_test_recall,high_test_map,
                    test_inverse_trans_pre, test_inverse_trans_label)
        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s"%(global_step,epoch),flush=True)
            
            print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                % (early_stop.best_rmse,early_stop.best_recall,early_stop.best_map),flush=True)
            print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                % (early_stop.best_high_rmse,early_stop.best_high_recall,early_stop.best_high_map),flush=True)
            torch.save(net.state_dict(),f'logs/best_{data_type}_{epoch:05d}.pth')
            break
    return early_stop.best_rmse,early_stop.best_recall,early_stop.best_map

def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    
    loaders = []
    scaler = ""
    train_data_shape = ""
    graph_feature_shape = ""
    for idx,(x,y,target_times,high_x,high_y,high_target_times,scaler) in enumerate(normal_and_generate_dataset_time(
                                    all_data_filename,
                                    train_rate=train_rate,
                                    valid_rate=valid_rate,
                                    recent_prior = recent_prior,
                                    week_prior = week_prior,
                                    one_day_period = one_day_period,
                                    days_of_week = days_of_week,
                                    pre_len = pre_len)):
        if args.test:
            x = x[:100]
            y = y[:100]
            target_times = target_times[:100]
            high_x = high_x[:100]
            high_y = high_y[:100]
            high_target_times = high_target_times[:100]

        if 'nyc' in all_data_filename:
            graph_x = x[:,:,[0,46,47],:,:].reshape((x.shape[0],x.shape[1],-1,north_south_map*west_east_map))
            high_graph_x = high_x[:,:,[0,46,47],:,:].reshape((high_x.shape[0],high_x.shape[1],-1,north_south_map*west_east_map))
        if 'chicago' in all_data_filename:
            graph_x = x[:,:,[0,39,40],:,:].reshape((x.shape[0],x.shape[1],-1,north_south_map*west_east_map))
            high_graph_x = high_x[:,:,[0,39,40],:,:].reshape((high_x.shape[0],high_x.shape[1],-1,north_south_map*west_east_map))


        print("feature:",str(x.shape),"label:",str(y.shape),"time:",str(target_times.shape),
            "high feature:",str(high_x.shape),"high label:",str(high_y.shape))
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape
            time_shape = target_times.shape
            graph_feature_shape = graph_x.shape
        loaders.append(Data.DataLoader(
            Data.TensorDataset(
                torch.from_numpy(x),
                torch.from_numpy(target_times),
                torch.from_numpy(graph_x),
                torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            ))
        if idx == 2:
            high_test_loader = Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(high_x),
                    torch.from_numpy(high_target_times),
                    torch.from_numpy(high_graph_x),
                    torch.from_numpy(high_y)
                    ),
                    batch_size=batch_size,
                    shuffle=(idx == 0)
                )
    train_loader, val_loader, test_loader = loaders


    ViT_Model =ViT(image_size=20, patch_size=5, num_classes=400, channels=7,
            dim=64, depth=6, heads=8, mlp_dim=128, data_type = config['data_type'])
    #multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!",flush=True)
        ViT_Model = nn.DataParallel(ViT_Model)
    ViT_Model.to(device)
    print(ViT_Model)

    num_of_parameters = 0
    for name,parameters in ViT_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)


    trainer = optim.Adam(ViT_Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience,delta=delta)
    
    risk_mask = get_mask(mask_filename)

    best_mae,best_mse,best_rmse = training(
            ViT_Model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            risk_mask,
            trainer,
            early_stop,
            device,
            scaler,
            data_type = config['data_type']
            )
    return best_mae,best_mse,best_rmse

if __name__ == "__main__":
    
    #python train_vit.py --config config/nyc/vit.json --gpus 0
    main(config)





