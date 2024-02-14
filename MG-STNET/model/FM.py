import torch
import torch.nn as nn
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


'''
import urllib3
import time
import webbrowser
while 1:
    # 创建一个连接池管理器
    time.sleep(1)
    http = urllib3.PoolManager()
    # 指定要访问的网址
    url = 'https://github.com/zouguojian'
    webbrowser.open(url)
'''


a = np.array([[7.3812,6.5207],[7.1304,6.3965],[7.1778,6.4402],[6.9931,6.2828],[7.0976,6.3971],
 [7.0019,6.2679],[7.0548,6.3236],[7.1288,6.4459],[7.0608,6.3864],[7.0053,6.2658]])
b = np.array([[34.00,34.46],[34.05,34.46],[34.38,34.18],[33.70,33.69],[34.31,34.32],
 [33.72,34.46],[34.31,34.11],[34.76,35.58],[32.98,34.01],[33.86,34.46]])
c = np.array([[0.1935,0.1807],[0.1895,0.1794],[0.1887,0.1776],[0.1886,0.1795],[0.1907,0.1824],
 [0.1863,0.1801],[0.1952,0.1860],[0.1933,0.1869],[0.1801,0.1735],[0.1875,0.1802]])

a1 = np.array([[10.9751,8.3253],[9.3907,6.7975],[9.1330,6.5649],[9.2540,6.8220],
 [9.3008,6.6804],[9.3724,6.6925],[9.5245,6.9606],[9.1195,6.4482],[10.8617,8.2382],[9.4456,7.0350]])
b1 = np.array([[20.87,22.09],[21.65,23.32],[21.71,22.50],[19.62,19.62],
 [21.05,22.50],[21.53,23.32],[18.96,18.93],[22.12,23.18],[19.80,20.85],[20.93,21.95]])
c1 = np.array([[0.0786,0.1013],[0.0904,0.1209],[0.0930,0.1132],[0.0883,0.0904],
 [0.0877,0.1137],[0.0933,0.1169],[0.0890,0.0958],[0.0985,0.1206],[0.0843,0.1047],[0.0980,0.1247]])

print((a[:-2]-a[7])/a[7] * 100)
print((b[:-2]-b[7])/b[7] * 100)
print((c[:-2]-c[7])/c[7] * 100)

print((a1[:-2]-a1[7])/a1[7] * 100)
print((b1[:-2]-b1[7])/b1[7] * 100)
print((c1[:-2]-c1[7])/c1[7] * 100)

# '''
# 创建要显示的数据
categories = ['W/o Mean-Std', 'W/o Bidirectional similarity matrices', 'W/o Adaptive graph', 'W/o Temporal Blocks',
              'W/o Adaptive channel fusion gate', 'W/o Geo-STModule', 'W/o MGraph-STModule', 'MG-STNET','GSNet']
values = list(a[:-1,0])
# categories = ['A', 'B', 'C', 'D', 'E']
# values = [4, 3, 5, 2, 1]

sns.set(font_scale=1.5)
# 计算数据在雷达图上的角度
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# 将第一个点和最后一个点连接起来，形成闭合图形
values += values[:1]
angles += angles[:1]
# 绘制雷达图
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.plot(angles, values, color='black')
# 填充雷达区域的颜色
ax.fill(angles, values, alpha=0.25)
# 设置刻度标签和刻度范围
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.yaxis.grid(True)
ax.set_theta_zero_location('N')                  #设置极轴方向
plt.grid(c='gray',linestyle='--',)   #设置网格线样式
plt.title('different kind',fontsize = 20)
plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0.0)) # 设置图例的位置，在画布外
# 设置雷达图的标题
ax.set_title("RMSE")
# 显示雷达图
plt.show()
# '''



class FM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FM, self).__init__()
        self.FC = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
                                nn.ReLU(),
                                nn.Conv2d(out_dim, out_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True))

    def forward(self, x, y):
        '''

        Args:
            x: [B, T, D, N, F]
            y: [B, T, D, N]

        Returns: [B, T, D, N]

        '''

        y = self.FC(y)
        x_1 = torch.sum(x, dim=-1)
        x_1 = torch.pow(x_1, exponent=2)  # [B, T, D, N]

        x_2 = torch.pow(x, exponent=2)
        x_2 = torch.sum(x_2, dim=-1) #  [B, T, D, N]

        return  0.5 * torch.subtract(x_1, x_2) + y
import numpy as np
# a = np.array([[10.3243,9.4994],[11.0165,10.1730],[8.3375, 7.3546],[7.9774,7.2806],
#      [7.9731,7.2750],[7.0608,6.3864],[7.0053,6.2658],[7.1288,6.4459]])
# b = np.array([[24.42,26.94],[23.14,25.22],[28.09,30.76],[30.81,31.22],
#      [30.42,31.43],[32.98,34.01],[33.86,34.46],[34.76,35.58]])
# c = np.array([[0.1049,0.1258],[0.1008,0.1119],[0.1228,0.1301],[0.1594,0.1536],
#      [0.1454,0.1498],[0.1801,0.1735],[0.1875,0.1802],[0.1933,0.1869]])

# print((a[5]-a[2])/a[2] * 100,(a[5]-a[3])/a[3] * 100,(a[5]-a[4])/a[4] * 100)
# print((b[5]-b[2])/b[2] * 100,(b[5]-b[3])/b[3] * 100,(b[5]-b[4])/b[4] * 100)
# print((c[5]-c[2])/c[2] * 100,(c[5]-c[3])/c[3] * 100,(c[5]-c[4])/c[4] * 100)

# print((a[6]-a[2])/a[2] * 100,(a[6]-a[3])/a[3] * 100,(a[6]-a[4])/a[4] * 100)
# print((b[6]-b[2])/b[2] * 100,(b[6]-b[3])/b[3] * 100,(b[6]-b[4])/b[4] * 100)
# print((c[6]-c[2])/c[2] * 100,(c[6]-c[3])/c[3] * 100,(c[6]-c[4])/c[4] * 100)

a=np.array([[14.9581,10.2564],[15.6946,10.3685],[12.6482,9.0421],[11.3382,8.7543],
   [11.3033,8.5437],[10.8617,8.2382],[9.4456,7.0350],[9.1195,6.4482]])
b=np.array([[13.80,15.89],[12.58,15.22],[17.83,18.66],[18.78,20.58],[18.43,18.93],
   [19.80,20.85],[20.93,21.95],[22.12,23.18]])
c=np.array([[0.0572,0.0644],[0.0545,0.0614],[0.0664,0.0758],[0.0753,0.1002],
   [0.0716,0.0770],[0.0843,0.1047],[0.0980,0.1247],[0.0985,0.1206]])

# print((a[5]-a[2])/a[2] * 100,(a[5]-a[3])/a[3] * 100,(a[5]-a[4])/a[4] * 100)
# print((b[5]-b[2])/b[2] * 100,(b[5]-b[3])/b[3] * 100,(b[5]-b[4])/b[4] * 100)
# print((c[5]-c[2])/c[2] * 100,(c[5]-c[3])/c[3] * 100,(c[5]-c[4])/c[4] * 100)

# print(np.round((a[7]-a)/a * 100, decimals=3))
# print(np.round((b[7]-b)/b * 100, decimals=3))
# print(np.round((c[7]-c)/c * 100, decimals=3))