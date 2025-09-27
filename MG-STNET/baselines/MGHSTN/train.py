import torch
import torch.optim as optim

import numpy as np
import json
from time import time
import argparse
import random

import sys
import os
from tqdm import tqdm

from lib.dataloader import get_mask, get_adjacent, get_grid_node_map_maxtrix, \
    get_trans, get_remote_sensing_dataloader, generate_dataloader
from lib.early_stop import EarlyStopping
from model.MGHSTN import MGHSTN
from lib.utils import mask_loss, compute_loss, predict_and_evaluate

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str, help="test program")
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--nors", action="store_true", help="no remote sensing")
parser.add_argument("--infer", action="store_true", help="load model from file and infer")

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def loadConfig(config):
    patience = config['patience']
    delta = config['delta']

    train_rate = config['train_rate']
    valid_rate = config['valid_rate']
    recent_prior = config['recent_prior']
    week_prior = config['week_prior']
    one_day_period = config['one_day_period']
    days_of_week = config['days_of_week']
    pre_len = config['pre_len']
    seq_len = recent_prior + week_prior
    training_epoch = config['training_epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_of_transformer_layers = config['num_of_transformer_layers']
    transformer_hidden_size = config['transformer_hidden_size']
    gcn_num_filter = config['gcn_num_filter']

    remote_sensing_image_path = config['remote_sensing_image_path']
    num_of_heads = config['num_of_heads']
    augment_channel = config['augment_channel']


    all_data_filename = []
    mask_filename = []
    road_adj_filename = []
    risk_adj_filename = []
    poi_adj_filename = []
    sum_adj_filename = []
    grid_node_filename = []
    north_south_map = []
    west_east_map = []


    trans20_10_filename = config['trans20_10_filename']
    trans10_5_filename = config['trans10_5_filename']
    trans5_2_filename = config['trans5_2_filename']

    trans = [trans20_10_filename, trans10_5_filename, trans5_2_filename]
    for i in range(4):
        all_data_filename.append(config['all_data_filename_{}'.format(i + 1)])
        mask_filename.append(config['mask_filename_{}'.format(i + 1)])
        road_adj_filename.append(config['road_adj_filename_{}'.format(i + 1)])
        risk_adj_filename.append(config['risk_adj_filename_{}'.format(i + 1)])
        poi_adj_filename.append(config['poi_adj_filename_{}'.format(i + 1)])
        sum_adj_filename.append(config['sum_adj_filename_{}'.format(i + 1)])
        grid_node_filename.append(config['grid_node_filename_{}'.format(i + 1)])
        north_south_map.append(config['north_south_map_{}'.format(i + 1)])
        west_east_map.append(config['west_east_map_{}'.format(i + 1)])

    bfc_20_10_filename = config['bfc_20_10_filename']


    return (patience, delta, train_rate, valid_rate, recent_prior, week_prior, one_day_period, days_of_week, pre_len,
            seq_len, training_epoch, batch_size, learning_rate, num_of_transformer_layers, transformer_hidden_size,
            gcn_num_filter, remote_sensing_image_path,
            all_data_filename, mask_filename, road_adj_filename, risk_adj_filename, poi_adj_filename, sum_adj_filename,
            grid_node_filename, north_south_map, west_east_map, trans, bfc_20_10_filename, num_of_heads, augment_channel)


def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             high_test_loader,
             road_adj,
             risk_adj,
             poi_adj,
             sum_adj,
             risk_mask,
             grid_node_map,
             trainer,
             early_stop,
             device,
             scaler,
             trans,
             bfc,
             data_type='nyc',
             ):

    global_step = 1
    for epoch in tqdm(range(1, training_epoch + 1)):
        net.train()
        epoch_iterator = tqdm(zip(train_loader[0],
                                  train_loader[1],
                                  train_loader[2],
                                  train_loader[3]),
                              desc=f"Epoch {epoch}/{training_epoch}",
                              unit="batch", leave=False)
        batch_num = 1
        for batch_1, batch_2, batch_3, batch_4 in epoch_iterator:
            batch = [batch_1, batch_2, batch_3, batch_4]
            train_feature = []
            target_time = []
            graph_feature = []
            train_label = []
            for i in range(4):
                t_train_feature, t_target_time, t_graph_feature, t_train_label = batch[i]
                t_train_feature, t_target_time, t_graph_feature, t_train_label = t_train_feature.to(device), \
                    t_target_time.to(device), t_graph_feature.to(device), t_train_label.to(device)
                train_feature.append(t_train_feature)
                target_time.append(t_target_time)
                graph_feature.append(t_graph_feature)
                train_label.append(t_train_label)

            final_output, classification_output = net(train_feature, target_time, graph_feature, road_adj, risk_adj,
                                                      poi_adj, grid_node_map, trans)
            l = mask_loss(final_output, classification_output, train_label, risk_mask, bfc, data_type)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            training_loss = l.cpu().item()

            batch_num += 1
            global_step += 1

            epoch_iterator.set_postfix({"Training Loss": f"{training_loss:.6f}"})

        # compute va/test loss
        val_loss = compute_loss(net, val_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                grid_node_map, trans, device, bfc, data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' % (global_step - 1, epoch, val_loss), flush=True)

        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                predict_and_evaluate(net, test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                     grid_node_map, trans, scaler, device)

            high_test_rmse, high_test_recall, high_test_map, _, _ = \
                predict_and_evaluate(net, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                     grid_node_map, trans, scaler, device)


            print('global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,'
                  'high test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
                  % (global_step - 1, epoch, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                     high_test_map), flush=True)

        # early stop according to val loss
        flag = early_stop(val_loss, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall, high_test_map,
                   test_inverse_trans_pre, test_inverse_trans_label)
        if flag:
            torch.save(net, data_type+'_full_model.pth')
            print("model saved in epoch: %s" % (epoch), flush=True)

        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s" % (global_step, epoch), flush=True)

            print('best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f'
                  % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map), flush=True)
            print('best test high RMSE: %.4f,best test high Recall: %.2f%%,best high test MAP: %.4f'
                  % (early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map), flush=True)
            break
    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map


def main(config):
    patience, delta, train_rate, valid_rate, recent_prior, week_prior, one_day_period, days_of_week, pre_len, \
        seq_len, training_epoch, batch_size, learning_rate, num_of_transformer_layers, transformer_hidden_size, \
        gcn_num_filter, remote_sensing_image_path, \
        all_data_filename, mask_filename, road_adj_filename, risk_adj_filename, poi_adj_filename, sum_adj_filename, grid_node_filename, \
        north_south_map, west_east_map, trans_filename, bfc_filename, num_of_heads, augment_channel = loadConfig(config)

    bfc = get_trans(bfc_filename)
    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    scaler = []
    train_data_shape = []
    time_shape = []
    graph_feature_shape = []
    high_test_loader = []
    train_loader = []
    val_loader = []
    test_loader = []
    grid_node_map = []

    for i in range(4):
        grid_node_map.append(get_grid_node_map_maxtrix(grid_node_filename[i]))
        t_scaler, t_train_data_shape, t_time_shape, t_graph_feature_shape, t_loaders, t_high_test_loader = \
            generate_dataloader(
                all_data_filename=all_data_filename[i],
                grid_node_data_filename=grid_node_filename[i],
                batch_size=batch_size,
                train_rate=train_rate,
                valid_rate=valid_rate,
                recent_prior=recent_prior,
                week_prior=week_prior,
                one_day_period=one_day_period,
                days_of_week=days_of_week,
                pre_len=pre_len,
                test=args.test,
                north_south_map=north_south_map[i],
                west_east_map=west_east_map[i]
            )
        scaler.append(t_scaler)
        train_data_shape.append(t_train_data_shape)
        time_shape.append(t_time_shape)
        graph_feature_shape.append(t_graph_feature_shape)
        high_test_loader.append(t_high_test_loader)

        t_train_loader, t_val_loader, t_test_loader = t_loaders
        train_loader.append(t_train_loader)
        val_loader.append(t_val_loader)
        test_loader.append(t_test_loader)

    trans = []

    for i in trans_filename:
        trans.append(get_trans(i))
    if not args.nors:
        rsdataloader = get_remote_sensing_dataloader('std', 256, remote_sensing_image_path)
        remote_sensing_data = next(iter(rsdataloader)).get('image').to(device)
    else:
        remote_sensing_data = None

    print("args.nors", args.nors)

    model = MGHSTN(train_data_shape[0][2], num_of_transformer_layers, seq_len, pre_len, transformer_hidden_size,
                   time_shape[0][1], graph_feature_shape[0][2], nums_of_filter, north_south_map, west_east_map,
                   args.nors, remote_sensing_data, num_of_heads, augment_channel)

    model.to(device)
    print(model)

    num_of_parameters = 0
    for name, parameters in model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience, delta=delta)

    risk_mask = []
    road_adj = []
    risk_adj = []
    poi_adj = []
    sum_adj = []
    for i in range(4):
        risk_mask.append(get_mask(mask_filename[i]))
        road_adj.append(get_adjacent(road_adj_filename[i]))
        risk_adj.append(get_adjacent(risk_adj_filename[i]))
        if poi_adj_filename[0] == "":
            poi_adj.append(None)
        else:
            poi_adj.append(get_adjacent(poi_adj_filename[i]))
        sum_adj.append(get_adjacent(sum_adj_filename[i]))

    for i in range(len(trans)):
        trans[i] = torch.from_numpy(trans[i]).unsqueeze(0).to(device)
    bfc = torch.from_numpy(bfc).unsqueeze(0).to(torch.float32).to(device)

    if args.infer:
        model = torch.load(config['data_type']+"_full_model.pth", map_location=device, weights_only=False)
        test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
            predict_and_evaluate(model, test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                 grid_node_map, trans, scaler, device)

        high_test_rmse, high_test_recall, high_test_map, _, _ = \
            predict_and_evaluate(model, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                 grid_node_map, trans, scaler, device)
        
        print('test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,'
              'high test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
              % (test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                 high_test_map), flush=True)
    else:
        training(
            model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            road_adj,
            risk_adj,
            poi_adj,
            sum_adj,
            risk_mask,
            grid_node_map,
            trainer,
            early_stop,
            device,
            scaler,
            trans,
            bfc,
            data_type=config['data_type'],
        )


if __name__ == "__main__":
    main(config)