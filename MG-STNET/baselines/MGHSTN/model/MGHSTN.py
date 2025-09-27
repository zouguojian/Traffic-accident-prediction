import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.RSencoder import ImageEncoder
from torch import Tensor

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term2 = torch.exp(torch.arange(0, d_model - 1, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term1)
        if d_model % 2 == 1:
            pe[:, 0, 1::2] = torch.cos(position * div_term2)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term1)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GCN_Layer(nn.Module):
    def __init__(self, num_of_features, num_of_filter):
        """One layer of GCN

        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features,
                      out_features=num_of_filter),
            nn.ReLU()
        )

    def forward(self, input, adj):
        """
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size, _, _ = input.shape  # 224，197，3
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size, 1, 1)  # 224，197，197
        input = torch.bmm(adj, input)  # 224，197，3
        output = self.gcn_layer(input)  # 224，197，64
        return output


class AdaptiveTemporalAttention(nn.Module):
    def __init__(self, d_model, num_layers, transformer_hidden_size, num_of_heads, num_of_target_time_feature, seq_len,
                 dropout=0.1):
        """
        Shared transformer encoder module for both grid and graph feature encoders.

        Arguments:
            d_model {int} -- The input dimension size for transformer.
            num_layers {int} -- The number of transformer encoder layers.
            transformer_hidden_size {int} -- The hidden size for transformer feedforward layers.
            num_of_heads {int} -- Number of heads in multi-head attention.
            num_of_target_time_feature {int} -- The number of target time feature (24 hour + 7 week + 1 holiday = 32).
            seq_len {int} -- The time length of the input sequence.
            dropout {float} -- Dropout rate. Default is 0.1.
        """
        super(AdaptiveTemporalAttention, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_of_heads,
                                       dim_feedforward=transformer_hidden_size * 4),
            num_layers=num_layers
        )
        self.fc0 = nn.Linear(in_features=d_model, out_features=transformer_hidden_size)

        # Attention layers for integrating target time features
        self.att_fc1 = nn.Linear(in_features=transformer_hidden_size, out_features=1)
        self.att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.att_bias = nn.Parameter(torch.zeros(1))
        self.att_softmax = nn.Softmax(dim=-1)

    def forward(self, input_features, target_time_feature, N):
        """
        Forward method for processing the input features with transformer encoder and attention mechanism.

        Arguments:
            input_features {Tensor} -- The input features to be encoded.
            target_time_feature {Tensor} -- The feature of target time, shape: (batch_size, num_target_time_feature).

        Returns:
            {Tensor} -- Encoded output, shape: (batch_size, hidden_size, W, H).
        """
        # Positional Encoding and Transformer Encoder
        x = input_features.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        transformer_output = self.fc0(transformer_output)

        # Attention Mechanism
        batch_size = target_time_feature.size(0)
        grid_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, N, 1).view(batch_size * N, -1)
        att_fc1_output = torch.squeeze(self.att_fc1(transformer_output))
        att_fc2_output = self.att_fc2(grid_target_time)
        att_score = self.att_softmax(F.relu(att_fc1_output + att_fc2_output + self.att_bias))
        att_score = att_score.view(batch_size * N, -1, 1)
        output = torch.sum(transformer_output * att_score, dim=1)

        return output


class RegionFeatureEncoder(nn.Module):
    def __init__(self, grid_in_channel, num_of_transformer_layers, seq_len,
                 transformer_hidden_size, num_of_target_time_feature, num_of_heads):
        """[summary]

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_transformer_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            transformer_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
            num_of_heads {int} -- the number of heads in multi-head attention
        """
        super(RegionFeatureEncoder, self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=grid_in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.ata = AdaptiveTemporalAttention(d_model=grid_in_channel, num_layers=num_of_transformer_layers,
                                             transformer_hidden_size=transformer_hidden_size,
                                             num_of_heads=num_of_heads,
                                             num_of_target_time_feature=num_of_target_time_feature,
                                             seq_len=seq_len)

    def forward(self, grid_input, target_time_feature):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
        Returns:
            {Tensor} -- shape：(batch_size,hidden_size,W,H)
        """
        batch_size, T, D, W, H = grid_input.shape

        grid_input = grid_input.view(-1, D, W, H)
        conv_output = self.grid_conv(grid_input)

        conv_output = conv_output.view(batch_size, -1, D, W, H) \
            .permute(0, 3, 4, 1, 2) \
            .contiguous() \
            .view(-1, T, D)

        grid_output = self.ata(conv_output, target_time_feature, W * H)
        grid_output = grid_output.view(batch_size, W, H, -1).permute(0, 3, 1, 2).contiguous()

        return grid_output


class GraphConv(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters, north_south_map, west_east_map):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data
        """
        super(GraphConv, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.road_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

    def forward(self, graph_feature, road_adj, risk_adj, poi_adj):

        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
        """
        batch_size, T, D1, N = graph_feature.shape

        road_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output, road_adj)

        risk_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output, risk_adj)

        graph_output = road_graph_output + risk_graph_output

        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output, poi_adj)
            graph_output += poi_graph_output

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1) \
            .view(batch_size, T, -1, N)

        return graph_output


class GraphTsEncoder(nn.Module):
    def __init__(self, seq_len, num_of_transformer_layers, transformer_hidden_size, num_of_target_time_feature,
                 north_south_map, west_east_map, num_of_heads):
        """
            seq_len {int} -- the time length of input
            num_of_transformer_layers {int} -- the number of GRU layers
            transformer_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
        """
        super(GraphTsEncoder, self).__init__()

        self.ata = AdaptiveTemporalAttention(d_model=64, num_layers=num_of_transformer_layers + 1,
                                             transformer_hidden_size=transformer_hidden_size,
                                             num_of_heads=num_of_heads,
                                             num_of_target_time_feature=num_of_target_time_feature,
                                             seq_len=seq_len)

        self.north_south_map = north_south_map
        self.west_east_map = west_east_map

    def forward(self, graph_output, target_time_feature, grid_node_map):
        """
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)
        """
        batch_size, T, _, N = graph_output.shape

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1)

        graph_output = graph_output.view(batch_size * N, T, -1)  # （32*197, 7, 64）

        graph_output = self.ata(graph_output, target_time_feature, N)

        graph_output = graph_output.view(batch_size, N, -1).contiguous()  # (32,197,64)

        grid_node_map_tmp = torch.from_numpy(grid_node_map) \
            .to(graph_output.device) \
            .repeat(batch_size, 1, 1)
        graph_output = torch.bmm(grid_node_map_tmp, graph_output) \
            .permute(0, 2, 1) \
            .view(batch_size, -1, self.north_south_map, self.west_east_map)
        return graph_output


class MGHSTN(nn.Module):
    def __init__(self, grid_in_channel, num_of_transformer_layers, seq_len, pre_len, transformer_hidden_size,
                 num_of_target_time_feature, num_of_graph_feature, nums_of_graph_filters, north_south_map,
                 west_east_map, is_nors, remote_sensing_data, num_of_heads, augment_channel):
        """[summary]

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_transformer_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            pre_len {int} -- the time length of prediction
            transformer_hidden_size {int} -- the hidden size of transformer
            num_of_target_time_feature {int} -- the number of target time feature，which is 24(hour)+7(week)+1(holiday)=32
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            north_south_map {list} -- the weight of grid data
            west_east_map {list} -- the height of grid data
            is_nors {bool} -- whether to use remote sensing data
            remote_sensing_data {Tensor} -- remote sensing data，shape：(batch_size,256,256)
            num_of_heads {int} -- the number of heads in multi-head attention
            augment_channel {int} -- the number of augment channel
        """
        super(MGHSTN, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.fusion_channel = 16
        self.augment_channel = augment_channel
        self.is_nors = is_nors

        if not self.is_nors:
            self.remote_sensing_data = remote_sensing_data
            self.RSencoder = ImageEncoder(256 * 256, self.augment_channel)
            self.aug = grid_in_channel + self.augment_channel
        else:
            self.aug = grid_in_channel
        self.RFEncoder = nn.ModuleList(
            [RegionFeatureEncoder(self.aug, num_of_transformer_layers, seq_len,
                                  transformer_hidden_size, num_of_target_time_feature,
                                  num_of_heads).cuda(),
             RegionFeatureEncoder(grid_in_channel, num_of_transformer_layers, seq_len,
                                  transformer_hidden_size, num_of_target_time_feature,
                                  num_of_heads).cuda(),
             RegionFeatureEncoder(grid_in_channel, num_of_transformer_layers, seq_len,
                                  transformer_hidden_size, num_of_target_time_feature,
                                  num_of_heads).cuda(),
             RegionFeatureEncoder(grid_in_channel, num_of_transformer_layers, seq_len,
                                  transformer_hidden_size, num_of_target_time_feature,
                                  num_of_heads).cuda()])
        self.GConv = nn.ModuleList([GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[0], west_east_map[0]).cuda(),
                                    GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[1], west_east_map[1]).cuda(),
                                    GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[2], west_east_map[2]).cuda(),
                                    GraphConv(num_of_graph_feature, nums_of_graph_filters,
                                              north_south_map[3], west_east_map[3]).cuda()])

        self.GTsEncoder = nn.ModuleList([GraphTsEncoder(seq_len, num_of_transformer_layers,
                                                        transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[0], west_east_map[0],
                                                        num_of_heads).cuda(),
                                         GraphTsEncoder(seq_len, num_of_transformer_layers,
                                                        transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[1], west_east_map[1],
                                                        num_of_heads).cuda(),
                                         GraphTsEncoder(seq_len, num_of_transformer_layers,
                                                        transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[2], west_east_map[2],
                                                        num_of_heads).cuda(),
                                         GraphTsEncoder(seq_len, num_of_transformer_layers,
                                                        transformer_hidden_size,
                                                        num_of_target_time_feature,
                                                        north_south_map[3], west_east_map[3],
                                                        num_of_heads).cuda()])

        self.grid_weight = nn.ModuleList([nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda(),
                                          nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda(),
                                          nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda(),
                                          nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda()])

        self.graph_weight = nn.ModuleList([nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda()
                                           ])

        self.output_layer = nn.ModuleList([nn.Linear(self.fusion_channel * north_south_map[0] * west_east_map[0],
                                                     pre_len * north_south_map[0] * west_east_map[0]).cuda(),
                                           nn.Linear(self.fusion_channel * north_south_map[1] * west_east_map[1],
                                                     pre_len * north_south_map[1] * west_east_map[1]).cuda(),
                                           nn.Linear(self.fusion_channel * north_south_map[2] * west_east_map[2],
                                                     pre_len * north_south_map[2] * west_east_map[2]).cuda(),
                                           nn.Linear(self.fusion_channel * north_south_map[3] * west_east_map[3],
                                                     pre_len * north_south_map[3] * west_east_map[3]).cuda()])

    def forward(self, grid_input, target_time_feature, graph_feature,
                road_adj, risk_adj, poi_adj, grid_node_map, trans):
        """
        Arguments:
            grid_input {list of Tensor} -- grid input，shape：(batch_size,T,D,W,H)
            graph_feature {list of Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            target_time_feature {list of Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            road_adj {list of np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {list of np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {list of np.array} -- poi adjacent matrix，shape：(N,N)
            grid_node_map {list of np.array} -- map graph data to grid data,shape (W*H,N)
            trans {list of np.array} -- the transformation matrix of graph data，shape：(n_c,n_f)
        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, _, _, _, _ = grid_input[0].shape

        grid_output = []
        graph_output = []
        fusion_output = []
        final_output = []
        classification_output = []

        if not self.is_nors:
            remote_output = self.RSencoder(self.remote_sensing_data)
            remote_output = remote_output.permute(1, 0).view(self.augment_channel, 20, 20).unsqueeze(0).unsqueeze(0) \
                .repeat(batch_size, 7, 1, 1, 1)
            grid_input[0] = torch.cat((grid_input[0], remote_output), dim=2)

        for i in range(4):
            t_grid_output = self.RFEncoder[i](grid_input[i], target_time_feature[i])
            t_graph_output = self.GConv[i](graph_feature[i], road_adj[i], risk_adj[i], poi_adj[i])
            t_grid_output = self.grid_weight[i](t_grid_output)

            grid_output.append(t_grid_output)
            graph_output.append(t_graph_output)

        # from fine to coarse
        for i in range(4 - 1):
            f_graph_output = graph_output[i]
            c_graph_output = graph_output[i + 1]

            batch_size, T, _, f_N = f_graph_output.shape
            batch_size1, T, _, c_N = c_graph_output.shape

            # coarse to fine
            c_graph_output = c_graph_output.reshape(batch_size1 * T, -1, c_N)
            cf_out = torch.matmul(c_graph_output, trans[i] / 3)
            f1_graph_output = f_graph_output + 0.2 * cf_out.reshape(batch_size1, T, -1, f_N)

            # fine to coarse
            f_graph_output = f_graph_output.reshape(batch_size * T, -1, f_N)
            fc_out = torch.matmul(f_graph_output, trans[i].permute(0, 2, 1) / 3)

            c_graph_output = c_graph_output.reshape(batch_size1, T, -1, c_N)
            c1_graph_output = c_graph_output + 0.8 * fc_out.reshape((batch_size, T, -1, c_N))

            graph_output[i] = f1_graph_output
            graph_output[i + 1] = c1_graph_output

        for i in range(4):
            graph_output[i] = self.GTsEncoder[i](graph_output[i], target_time_feature[i], grid_node_map[i])
            graph_output[i] = self.graph_weight[i](graph_output[i])
            fusion_output.append(grid_output[i] + graph_output[i])  # 16,20,20

        for i in range(4):
            fusion_output[i] = fusion_output[i].view(batch_size, -1)
            final_output.append(self.output_layer[i](fusion_output[i])
                                .view(batch_size, -1, self.north_south_map[i], self.west_east_map[i]))

            classification_output.append(torch.relu(final_output[i].view(final_output[i].shape[0], -1)))

        return final_output, classification_output
