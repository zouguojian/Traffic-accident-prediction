import torch.nn as nn
import torch
import numpy as np

class Embedding(nn.Module):
    def __init__(self, in_dim = 64, out_dim = 64):
        '''

        Args:
            in_dim:
            out_dim:
        '''
        super(Embedding,self).__init__()
        self.FC = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
                                nn.ReLU(),
                                nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True))

    def forward(self, x):
        '''

        Args:
            x: [B * T, D1, N, D]

        Returns:

        '''
        return self.FC(x)

class EmbeddingS(nn.Module):
    '''
    spatio-temporal embedding
    TE:     [batch_size, T, D]
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, in_dims = [], out_dim = 64):
        super(EmbeddingS, self).__init__()
        self.in_dims =in_dims
        self.AllE = nn.ModuleList()
        for dim in in_dims:
            self.AllE.append(Embedding(in_dim=dim, out_dim=out_dim))

    def forward(self, Features):
        '''

        Args:
            Features: [B * T, D, N, D1]
            D: 表示初试各个特征所在的维度，如NYC为 [1, 24, 7, 1]+ [1] * 7 + [1, 1, 1, 1, 1, 1, 1, 1]，
            D1: 最终的特征种类，对于NYC，为19
        Returns:

        '''
        newFeatures=[]
        current_index = 0
        for i, index_range in enumerate(self.in_dims):
            x = torch.unsqueeze(Features[:, current_index: current_index+index_range], dim=3)
            x = self.AllE[i](x)
            newFeatures.append(x)
            current_index += index_range
        return torch.cat(newFeatures, dim=-1)

x = [-0.17201234, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.32656217, 0.06356036, 0.13170053, 0.698475, 0.22203368, 0.5129088, -0.15624946, -1.2833313, 1.0, 0.0, 0.0, 0.0, 0.0, 0.40462983, 0.12406088]
fields=[1, 24, 7, 1]+ [1] * 7 + [1, 1, 1, 1, 1, 1, 1, 1]
x = np.array(x)
x = np.expand_dims(x,[0, 1, 3]) # [B, T, fields, N]
B, T, D, N = x.shape
x = np.reshape(x, [B * T, D, N])
print(x.shape)
print(fields)
emb = EmbeddingS(fields, out_dim= 64)
x = emb(torch.from_numpy(np.array(x, dtype=np.float32)))
print(x.shape)