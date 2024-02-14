import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from model.TCN import TemporalConvNet
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def compute_cosine_distances(a, b):
    # x shape [B, N, D]
    # y shape [B, N, D]
    # results shape [B, N, N]

    a, b = normalize(a), normalize(b)

    dis = 1 - torch.matmul(a, b.transpose(2, 1))

    return dis


class Embedding(nn.Module):
    '''
    spatio-temporal embedding
    TE:     [batch_size, T, D]
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, time_futures, D):
        super(Embedding, self).__init__()
        self.FC_te = nn.Sequential(
            nn.Conv2d(time_futures, D, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(D, D, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True))

    def forward(self, TE):
        '''
        Args:
            TE: [B, T, D*]
        Returns: [B, T, D]

        '''
        # temporal embedding
        TE = torch.unsqueeze(TE, dim=3)
        TE = self.FC_te(TE.transpose(2, 1))
        return TE.transpose(2, 1)


class Nconv(nn.Module):
    def __init__(self):
        super(Nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class AdaptiveGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(AdaptiveGCN, self).__init__()
        self.nconv = Nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):  # [B, D, N, T]
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        # B, D, N, T = x.shape
        # x = x.transpose(3, 1).contiguous().view(B * T, N, D) # [B * T, N, D]
        # cos_dis = compute_cosine_distances(a=x, b=x) # [B * T, N, N]
        # x = torch.matmul(cos_dis, x) # [B * T, N, D]
        # x = x.view(B, T, N, D).transpose(3, 1)  # [B, D, N, T]
        # out.append(x)

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, D, num_vertex, num_step]
    HT:     [batch_size, D, num_vertex, num_step]
    D:      output dims
    return: [batch_size, D, num_vertex, num_step]
    '''

    def __init__(self, D, bn_decay=0.2):
        super(gatedFusion, self).__init__()

        self.FC_xs = nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=False)
        self.FC_xt = nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=True)
        self.FC_h = nn.Sequential(nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1, bias=True))

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STGeoModule(nn.Module):
    def __init__(self, grid_in_channel, seq_len, emb_size, num_of_target_time_feature):
        """[summary]

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            seq_len {int} -- the time length of input
            emb_size {int} -- the hidden size of feature
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
        """
        super(STGeoModule, self).__init__()
        self.seq_len = seq_len
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=grid_in_channel, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.tcn = TemporalConvNet(grid_in_channel, [emb_size, emb_size, emb_size], kernel_size=2, dropout=0.)
        # self.attention = transformAttention(K=8, d = emb_size//8)

    def forward(self, grid_input):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size, T, D, 1, 1)
        Returns:
            {Tensor} -- shape：(batch_size,hidden_size,W,H)
        """
        batch_size, T, D, W, H = grid_input.shape
        grid_input = grid_input.view(-1, D, W, H)
        conv_output = self.grid_conv(grid_input)  # (B * T, D, W, H)

        grid_output = self.tcn(conv_output.view(batch_size, T, D, W, H)).contiguous().view(batch_size * W * H, -1,
                                                                                           64)  # [-1, T, D]  # [B, W * H, T, D]
        # .view(batch_size * W * H, -1, 64)  # [-1, T, D]
        # gru_output = self.attention(gru_output, target_time_feature[:,:,:self.seq_len], target_time_feature[:,:,self.seq_len:])  # [B, T, N, D]

        grid_output = grid_output[:, -1].view(batch_size, W, H, -1).permute(0, 3, 1, 2).contiguous()

        return grid_output


class STSemModule(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters,
                 seq_len, emb_size,
                 north_south_map, west_east_map,
                 support_lists, adjinit_lists, device, gcn_bool=True, addaptadj=True, dropout=0.):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            seq_len {int} -- the time length of input
            emb_size {int} -- the hidden size of the feature
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data

        """
        super(STSemModule, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.seq_len = seq_len
        self.gcn_bool = gcn_bool
        self.support_lists = support_lists
        self.supports_lens = [0, 0, 0]
        for i in range(len(support_lists)):
            if support_lists[i] is not None:
                self.supports_lens[i] += len(support_lists[i])

        N = adjinit_lists[0].shape[0]
        if gcn_bool and addaptadj:
            self.nodevec1_list = []
            self.nodevec2_list = []
            for i, adjinit in enumerate(adjinit_lists):
                if adjinit is None:
                    if support_lists[i] is None:
                        self.support_lists[i] = []
                    self.nodevec1_list.append(
                        nn.Parameter(torch.randn(N, 10).to(device), requires_grad=True).to(device))
                    self.nodevec2_list.append(
                        nn.Parameter(torch.randn(10, N).to(device), requires_grad=True).to(device))
                    self.supports_lens[i] += 1
                else:
                    if support_lists[i] is None:
                        self.support_lists[i] = []
                    m, p, n = torch.svd(adjinit)
                    # print(m.shape, p.shape, n.shape)
                    initemb1 = torch.mm(m[:, :100], torch.diag(p[:100] ** 0.5))
                    initemb2 = torch.mm(torch.diag(p[:100] ** 0.5), n[:, :100].t())
                    self.nodevec1_list.append(nn.Parameter(initemb1, requires_grad=True).to(device))
                    self.nodevec2_list.append(nn.Parameter(initemb2, requires_grad=True).to(device))
                    self.supports_lens[i] += 1
        self.layers = len(nums_of_graph_filters)
        self.road_gcn = nn.ModuleList()
        self.risk_gcn = nn.ModuleList()
        self.poi_gcn = nn.ModuleList()
        if gcn_bool:
            for layer in range(self.layers):
                if layer == 0:
                    self.road_gcn.append(AdaptiveGCN(num_of_graph_feature, nums_of_graph_filters[layer], dropout,
                                                     support_len=self.supports_lens[0]))
                    self.risk_gcn.append(AdaptiveGCN(num_of_graph_feature, nums_of_graph_filters[layer], dropout,
                                                     support_len=self.supports_lens[1]))
                    self.poi_gcn.append(AdaptiveGCN(num_of_graph_feature, nums_of_graph_filters[layer], dropout,
                                                    support_len=self.supports_lens[2]))
                else:
                    self.road_gcn.append(
                        AdaptiveGCN(nums_of_graph_filters[layer - 1], nums_of_graph_filters[layer], dropout,
                                    support_len=self.supports_lens[0]))
                    self.risk_gcn.append(
                        AdaptiveGCN(nums_of_graph_filters[layer - 1], nums_of_graph_filters[layer], dropout,
                                    support_len=self.supports_lens[1]))
                    self.poi_gcn.append(
                        AdaptiveGCN(nums_of_graph_filters[layer - 1], nums_of_graph_filters[layer], dropout,
                                    support_len=self.supports_lens[2]))
        # self.concat_layer = nn.Sequential(nn.Conv2d(in_channels=emb_size * 3, out_channels=emb_size, kernel_size=(1, 1)),
        #                                     nn.ReLU(),
        #                                     nn.Conv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 1)))
        self.tcn = TemporalConvNet(nums_of_graph_filters[-1], [emb_size, emb_size, emb_size], kernel_size=2, dropout=0.)
        # self.attention = transformAttention(K=8, d = emb_size//8)

    def forward(self, graph_feature, grid_node_map):
        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size, T, D, 1)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)
        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, T, D, N = graph_feature.shape
        new_supports = [None, None, None]
        if self.gcn_bool:
            for i in range(len(self.support_lists)):
                adp = F.softmax(F.relu(torch.mm(self.nodevec1_list[i], self.nodevec2_list[i])), dim=1)
                new_supports[i] = (self.support_lists[i] + [adp]) if self.support_lists[i] else [adp]
        graph_output = graph_feature.permute(0, 2, 3, 1).contiguous()  # [B, D, N, T]
        final_output = []
        # print(len(new_supports))
        # for i in range(len(new_supports)):
        #     pd.to_pickle(new_supports[i][-1].numpy(),'adaptive'+str(i+1)+'.pkl')
        for layer in range(self.layers):
            road_graph_output = self.road_gcn[layer](graph_output, new_supports[0])
            risk_graph_output = self.risk_gcn[layer](graph_output, new_supports[1])
            poi_graph_output = self.poi_gcn[layer](graph_output, new_supports[2])
            # graph_output = self.concat_layer(torch.cat([road_graph_output, risk_graph_output, poi_graph_output],  dim=1))
            graph_output = road_graph_output + risk_graph_output + poi_graph_output

        graph_output = torch.unsqueeze(graph_output, dim=4).permute(0, 3, 1, 2, 4)  # [B, T, D, N, 1]
        graph_output = self.tcn(graph_output).contiguous().view(batch_size * N, -1, 64)  # [B, N, T, D]
        # .view(batch_size * N, -1, 64)    # [B * N, -1, D]
        # graph_output = self.attention(graph_output, target_time_feature[:,:,:self.seq_len], target_time_feature[:,:,self.seq_len:])  # [B, T, N, D]
        graph_output = graph_output[:, -1].view(batch_size, N, -1).contiguous()  # [B, N, D]

        grid_node_map_tmp = torch.from_numpy(grid_node_map) \
            .to(graph_feature.device) \
            .repeat(batch_size, 1, 1)
        graph_output = torch.bmm(grid_node_map_tmp, graph_output) \
            .permute(0, 2, 1) \
            .view(batch_size, -1, self.north_south_map, self.west_east_map)

        return graph_output


class M2STN(nn.Module):
    def __init__(self, grid_in_channel, seq_len, pre_len,
                 emb_size, num_of_target_time_feature,
                 num_of_graph_feature, nums_of_graph_filters,
                 north_south_map, west_east_map,
                 support_lists, adjinit_lists, device, gcn_bool=True, addaptadj=True):
        """[summary]

        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            seq_len {int} -- the time length of input
            pre_len {int} -- the time length of prediction
            emb_size {int} -- the hidden size of the feature
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data
        """
        super(M2STN, self).__init__()
        self.emb_size = emb_size
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.timeEmbedding = Embedding(time_futures=num_of_target_time_feature, D=emb_size)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(grid_in_channel, emb_size, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(emb_size, emb_size, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_of_graph_feature, emb_size, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(emb_size, emb_size, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True))

        self.st_geo_module = STGeoModule(self.emb_size, seq_len, emb_size, num_of_target_time_feature)

        self.st_sem_module = STSemModule(self.emb_size, nums_of_graph_filters,
                                         seq_len, emb_size,
                                         north_south_map, west_east_map,
                                         support_lists, adjinit_lists, device, gcn_bool=gcn_bool, addaptadj=addaptadj)

        self.gate_f = gatedFusion(emb_size)

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=emb_size, out_channels=emb_size, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=emb_size, out_channels=1, kernel_size=(1, 1)))

    def forward(self, grid_input, target_time_feature, graph_feature,
                grid_node_map):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,T,D,W,H)
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)

        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        time_emb = self.timeEmbedding(target_time_feature)  # [B, T, D ,1]
        grid_input = torch.cat([grid_input, grid_input[:, 3:4]], dim=1)
        B, T, D, W, H = grid_input.shape
        grid_input = self.conv_1(grid_input.view(B * T, D, W, H)).view(B, T, self.emb_size, W, H)
        grid_output = self.st_geo_module(grid_input + torch.unsqueeze(time_emb, dim=4))

        graph_feature = torch.cat([graph_feature, graph_feature[:, 3:4]], dim=1)
        B, T, D, N = graph_feature.shape
        graph_feature = self.conv_2(torch.unsqueeze(graph_feature.view(B * T, D, N), dim=3)).view(B, T, self.emb_size,
                                                                                                  N)
        graph_output = self.st_sem_module(graph_feature + time_emb, grid_node_map)

        final_output = self.gate_f(grid_output, graph_output)
        final_output = self.output_layer(final_output)
        return final_output