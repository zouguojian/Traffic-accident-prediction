# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np


font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11,
}



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vis_fun(pre = None, pre1 = None, pre2 = None, obs = None):
    '''

    Args:
        pre: predicted values [N, N]
        obs: observed values [N, N]

    Returns:

    '''
    f, ax = plt.subplots(figsize=(30, 10))
    plt.subplot(1, 4, 1)
    sns.set(font_scale=1.)
    hm = sns.heatmap(pre,
                     cbar=False,
                     annot=True,
                     square=True,
                     # fmt=".1f",
                     linewidths=.5,
                     cmap="Blues",  # 刻度颜色
                     annot_kws={"size": 10},
                     xticklabels=[i for i in range(1, 21)],
                     yticklabels=[i for i in range(1, 21)])  # seaborn.heatmap相关属性
    # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.ylabel(fontsize=15,)
    # plt.xlabel(fontsize=15)
    plt.title("MG-STNET", fontsize=20)

    plt.subplot(1, 4, 2)
    sns.set(font_scale=1.)
    hm = sns.heatmap(pre1,
                     cbar=False,
                     annot=True,
                     square=True,
                     # fmt=".1f",
                     linewidths=.5,
                     cmap="Blues",  # 刻度颜色
                     annot_kws={"size": 10},
                     xticklabels=[i for i in range(1, 21)],
                     yticklabels=[i for i in range(1, 21)])  # seaborn.heatmap相关属性
    # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.ylabel(fontsize=15,)
    # plt.xlabel(fontsize=15)
    plt.title("GSNet", fontsize=20)

    plt.subplot(1, 4, 3)
    sns.set(font_scale=1.)
    hm = sns.heatmap(pre2,
                     cbar=False,
                     annot=True,
                     square=True,
                     # fmt=".1f",
                     linewidths=.5,
                     cmap="Blues",  # 刻度颜色
                     annot_kws={"size": 10},
                     xticklabels=[i for i in range(1, 21)],
                     yticklabels=[i for i in range(1, 21)])  # seaborn.heatmap相关属性
    # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.ylabel(fontsize=15,)
    # plt.xlabel(fontsize=15)
    plt.title("C-ViT", fontsize=20)

    plt.subplot(1, 4, 4)
    sns.set(font_scale=1.)
    hm = sns.heatmap(obs,
                     cbar=False,
                     annot=True,
                     square=True,
                     # fmt=".1f",
                     linewidths=.5,
                     cmap="Blues",  # 刻度颜色
                     annot_kws={"size": 10},
                     xticklabels= [i for i in range(1, 21)],
                     yticklabels=[i for i in range(1, 21)])  # seaborn.heatmap相关属性
    # 解决中文显示问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("Ground truth", fontsize=20)
    plt.show()
    return

# Chi_pre = np.load('MGSTN-chicago.npz')['prediction']
# Chi_obs = np.load('MGSTN-chicago.npz')['truth']
# print(Chi_pre.shape, Chi_obs.shape)
# for i in range(Chi_pre.shape[0]):
#     vis_fun(pre=Chi_pre[i,0], obs=Chi_obs[i,0])

def mask_func(data):
    zeros= np.zeros(shape=[400])
    # 对数组进行降序排列并返回索引值
    data = np.reshape(data, newshape=[400])
    sorted_indices = np.argsort(-data)[:100]

    # 根据索引值从原始数组中提取前k个最大的元素
    zeros[sorted_indices] = 1

    return np.reshape(zeros, [20, 20])

# NYC_pre = np.load('MGSTN-nyc.npz')['prediction']
# NYC_obs = np.load('MGSTN-nyc.npz')['truth']
# NYC_pre1 = np.load('GSNet-nyc.npz')['prediction']
# NYC_pre2 = np.load('C-ViT-nyc.npz')['prediction']

NYC_pre = np.load('MGSTN-chicago.npz')['prediction']
NYC_obs = np.load('MGSTN-chicago.npz')['truth']
NYC_pre1 = np.load('GSNet-chicago.npz')['prediction']
NYC_pre2 = np.load('C-ViT-chicago.npz')['prediction']

# mask = pd.read_pickle('risk_mask.pkl')
print(NYC_pre.shape, NYC_pre1.shape, NYC_pre2.shape, NYC_obs.shape)

# 天
# '''
hour = 9
week = 6
for i in range(NYC_pre.shape[0]):
    if (hour + 1) % 24 == 0:
        hour = 0
        if (week+1) > 7:
            week =1
        else: week+=1
    else: hour+=1
    print('hour is : ', hour, ' week is :', week)
    # mask = np.where(NYC_pre[i,0] > 0.5, 1, 0).astype('float32')
    mask = mask_func(NYC_pre[i,0])
    mask1 = mask_func(NYC_pre1[i, 0])
    mask2 = mask_func(NYC_pre2[i, 0])
    a, b, c = np.round(np.multiply(NYC_pre[i,0], mask), decimals=1),np.round(np.multiply(NYC_pre1[i,0], mask1), decimals=1),np.round(np.multiply(NYC_pre2[i,0], mask2), decimals=1)
    # print(a[9], b[9], c[9])
    # print(np.sum(a),np.sum(b),np.sum(c))
    vis_fun(pre=np.round(np.multiply(NYC_pre[i,0], mask), decimals=1),
            pre1=np.round(np.multiply(NYC_pre1[i,0], mask1), decimals=1),
            pre2=np.round(np.multiply(NYC_pre2[i,0], mask2), decimals=1), obs=NYC_obs[i,0])
# '''

'''
# 一周
hour = 9
week = 6
for i in range(NYC_pre.shape[0]):
    if (hour + 1) % 24 == 0:
        hour = 0
        if (week+1) > 7:
            week =1
        else: week+=1
    else: hour+=1
    print('hour is : ', hour, ' week is :', week)
    # mask = np.where(NYC_pre[i,0] > 0.5, 1, 0).astype('float32')
    mask = mask_func(NYC_pre[[i+j*24 for j in range(7)],0].sum(axis=0))
    mask1 = mask_func(NYC_pre1[[i+j*24 for j in range(7)], 0].sum(axis=0))
    mask2 = mask_func(NYC_pre2[[i+j*24 for j in range(7)], 0].sum(axis=0))
    a, b, c = np.round(np.multiply(NYC_pre[i,0], mask), decimals=1),np.round(np.multiply(NYC_pre1[i,0], mask1), decimals=1),np.round(np.multiply(NYC_pre2[i,0], mask2), decimals=1)
    # print(a[9], b[9], c[9])
    # print(np.sum(a),np.sum(b),np.sum(c))
    vis_fun(pre=np.round(np.multiply(NYC_pre[[i+j*24 for j in range(7)],0].sum(axis=0), mask), decimals=1),
            pre1=np.round(np.multiply(NYC_pre1[[i+j*24 for j in range(7)],0].sum(axis=0), mask1), decimals=1),
            pre2=np.round(np.multiply(NYC_pre2[[i+j*24 for j in range(7)],0].sum(axis=0), mask2), decimals=1), obs=NYC_obs[[i+j*24 for j in range(7)],0].sum(axis=0))
'''

data = pd.read_pickle('/Users/zouguojian/Traffic-accident-risk/data/nyc/poi_adj.pkl')
# f, ax = plt.subplots(figsize=(22, 13))
print(data.shape)
sns.set(font_scale=1.)
hm = sns.heatmap(data,
                 cbar=False,
                 annot=False,
                 square=True,
                 fmt=".2f",
                 linewidths=.5,
                 cmap="RdPu",  # 刻度颜色
                 annot_kws={"size": 10},
                 xticklabels=[i for i in range(1, 244)],
                 yticklabels=[i for i in range(1, 244)])  # seaborn.heatmap相关属性
# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.title("Similarity matrix", fontsize=20)
plt.show()
