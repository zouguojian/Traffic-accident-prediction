import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, feature_size, d_model):
        super(ImageEncoder, self).__init__()
        self.feature_size = feature_size//16

        self.conv_1 = nn.Conv2d(3, 8, 1, 1, 0)
        self.relu_1 = nn.ReLU(inplace=True)
        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128

        self.conv_2 = nn.Conv2d(8, 8, 3, 1, 1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)# 64

        self.conv_3 = nn.Conv2d(8, 1, 3, 1, 1)
        self.relu_3 = nn.ReLU(inplace=True) # 1, 64, 64

        self.linear = nn.Linear(self.feature_size, d_model)

    def forward(self, input):
        output = self.conv_1(input)
        output = self.relu_1(output)
        output = self.maxPool_1(output)

        output = self.conv_2(output)
        output = self.relu_2(output)
        output = self.maxPool_2(output)

        output = self.conv_3(output)
        output = self.relu_3(output)

        output = self.linear(output.view(-1, self.feature_size))
        return self.normalize(output)

    def normalize(self, input):
        mean = input.mean(dim=0).unsqueeze(0)
        std = input.std(dim=0).unsqueeze(0)
        return (input-mean)/std

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32*16*16, out_features=256)

        self.fc2 = nn.Linear(in_features=256, out_features=16)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool4(x)

        x = x.view(-1, 32*16*16)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
