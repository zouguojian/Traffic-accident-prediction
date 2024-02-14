import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

import numpy as np
import json
from time import time
from datetime import datetime
import argparse
import random

import sys
import os

'''
这里我们将纽约的数据作为案例进行注释，以便于读者的理解和可复线
'''

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time, get_mask, get_adjacent, get_grid_node_map_maxtrix
from model.M2STN import M2STN
from lib.utils import mask_loss, compute_loss, predict_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str, help="test program")
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly', type=bool, default=False, help='whether only adaptive adj')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')
parser.add_argument('--randomadj', type=bool, default=True, help='whether random initialize adaptive adj')

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']    # (8760, 48, 20, 20)
mask_filename = config['mask_filename']            # (20, 20)

road_adj_filename = config['road_adj_filename']    # (243, 243)
risk_adj_filename = config['risk_adj_filename']    # (243, 243)
poi_adj_filename = config['poi_adj_filename']      # (243, 243)
grid_node_filename = config['grid_node_filename']  # (400, 243)
grid_node_map = get_grid_node_map_maxtrix(grid_node_filename)  # 网格到有效具有路网节点的映射
num_of_vertices = grid_node_map.shape[1]  # 实际包含的路网有效区域

patience = config['patience']
delta = config['delta']

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

'''
基础参数，包括数据的划分比率、数据特征尺度、epoch等信息
'''
train_rate = config['train_rate']
valid_rate = config['valid_rate']
recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior  # 一条数据样本的时间步长
training_epoch = config['training_epoch']


def train(model,
          training_epoch,
          train_loader,
          val_loader,
          risk_mask,
          trainer,
          device,
          data_type='nyc',
          config=None):
    global_step = 1
    min_loss = 1000000.0
    patience = 0
    for epoch in range(training_epoch):
        model.train()
        batch = 1
        for train_feature, target_time, gragh_feature, train_label in train_loader:
            start_time = time()
            train_feature, target_time, gragh_feature, train_label = train_feature.to(device), target_time.to(
                device), gragh_feature.to(device), train_label.to(device)
            l = mask_loss(model(train_feature, target_time, gragh_feature, grid_node_map),
                          train_label, risk_mask, data_type=data_type)  # l的shape：(1,)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            training_loss = l.cpu().item()

            print('global step: %s, epoch: %s, batch: %s, training loss: %.6f, time: %.2fs'
                  % (global_step, epoch + 1, batch, training_loss, time() - start_time), flush=True)
            batch += 1
            global_step += 1

        # compute validation loss
        val_loss = validation(model, val_loader, risk_mask, grid_node_map, device, data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' % (global_step - 1, epoch + 1, val_loss), flush=True)

        if val_loss < min_loss:
            print('in the %dth epoch, val loss decrease from %.6f to %.6f, saving to %s' % (
            epoch + 1, min_loss, val_loss, config['model_file']))
            min_loss = val_loss
            patience = 0
            torch.save(model, config['model_file'])
        else:
            patience += 1
            print(f'EarlyStopping counter: {patience} out of %d' % config['patience'])
            if patience > config['patience']:
                break
    return

def validation(model,
               val_loader,
               risk_mask,
               grid_node_map,
               device,
               data_type):
    val_loss = compute_loss(model, val_loader, risk_mask, grid_node_map, device, data_type)
    return val_loss


def test(test_loader,
         high_test_loader,
         risk_mask,
         device,
         scaler,
         config=None):
    model = torch.load(config['model_file'],map_location=device)
    test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
        predict_and_evaluate(model, test_loader, risk_mask, grid_node_map,
                             scaler, device)
    high_test_rmse, high_test_recall, high_test_map, _, _ = \
        predict_and_evaluate(model, high_test_loader, risk_mask, grid_node_map,
                             scaler, device)
    print('                RMSE\t\tRecall\t\tMAP')
    print('Test            %.4f\t\t%.2f%%\t\t%.4f' % (test_rmse, test_recall, test_map))
    print('High test       %.4f\t\t%.2f%%\t\t%.4f' % (high_test_rmse, high_test_recall, high_test_map))

    # np.savez_compressed('results/MGSTN-' + 'chicago', **{'prediction': test_inverse_trans_pre, 'truth': test_inverse_trans_label})


def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_of_gru_layers = config['num_of_gru_layers']
    gru_hidden_size = config['gru_hidden_size']
    gcn_num_filter = config['gcn_num_filter']

    loaders = []
    train_data_shape = ""
    graph_feature_shape = ""
    for idx, (x, y, target_times, high_x, high_y, high_target_times, scaler) in enumerate(
            normal_and_generate_dataset_time(
                    all_data_filename,
                    train_rate=train_rate,
                    valid_rate=valid_rate,
                    recent_prior=recent_prior,
                    week_prior=week_prior,
                    one_day_period=one_day_period,
                    days_of_week=days_of_week,
                    pre_len=pre_len)):

        if 'nyc' in all_data_filename:
            graph_x = x[:, :, [0, 46, 47], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 46, 47], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)
        if 'chicago' in all_data_filename:
            graph_x = x[:, :, [0, 39, 40], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 39, 40], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)

        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape),
              "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))
        '''
        feature: (4584, 7, 48, 20, 20) label: (4584, 1, 20, 20) time: (4584, 32) high feature: (1337, 7, 48, 20, 20) high label: (1337, 1, 20, 20)
        graph_x: (4584, 7, 3, 243) high_graph_x: (1337, 7, 3, 243)
        '''
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

    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)
    risk_mask = get_mask(mask_filename)
    road_adj = get_adjacent(road_adj_filename)
    road_supports = [torch.tensor(i).to(device) for i in road_adj]
    risk_adj = get_adjacent(risk_adj_filename)
    risk_supports = [torch.tensor(i).to(device) for i in risk_adj]
    if poi_adj_filename == "":
        poi_adj = None
    else:
        poi_adj = get_adjacent(poi_adj_filename)
    poi_supports = [torch.tensor(i).to(device) for i in poi_adj] if poi_adj else None
    if args.randomadj:
        risk_adjinit, road_adjinit, poi_adjinit = risk_supports[0], road_supports[0], poi_supports[0] if poi_adj else None
    else:
        risk_adjinit, road_adjinit, poi_adjinit = None, None, None

    if args.aptonly:
        road_supports, risk_supports, poi_supports = None, None, None

    M2STN_Model = M2STN(train_data_shape[2], seq_len, pre_len,
                        gru_hidden_size, time_shape[2], graph_feature_shape[2],
                        nums_of_filter, north_south_map, west_east_map,
                        support_lists=[road_supports, risk_supports, poi_supports],
                        adjinit_lists=[road_adjinit, risk_adjinit, poi_adjinit], device=device, gcn_bool=args.gcn_bool,
                        addaptadj=args.addaptadj)
    # multi gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        GSNet_Model = nn.DataParallel(M2STN_Model)
    M2STN_Model.to(device)
    print(M2STN_Model)

    num_of_parameters = 0
    for name, parameters in M2STN_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(M2STN_Model.parameters(), lr=learning_rate)

    if not args.test:  # training
        train(M2STN_Model,
              training_epoch,
              train_loader,
              val_loader,
              risk_mask,
              trainer,
              device,
              data_type=config['data_type'],
              config=config
              )
    else:  # testing
        test(test_loader,
             high_test_loader,
             risk_mask,
             device,
             scaler,
             config=config)


if __name__ == "__main__":
    # python train.py --config config/nyc/GSNet_NYC_Config.json --gpus 0
    main(config)