import torch.nn as nn


class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(3, 16, 5, 2, 0)
        self.bn1_1 = nn.BatchNorm2d(16)
        # block 2
        self.conv2_1 = nn.Conv2d(16, 32, 3, 1, 0)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 16, 3, 1, 0)
        self.bn2_2 = nn.BatchNorm2d(16)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.bn3_1 = nn.BatchNorm2d(24)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        self.bn3_2 = nn.BatchNorm2d(24)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(40)
        # points branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(80)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.bn_ip1 = nn.BatchNorm1d(128)
        self.ip2 = nn.Linear(128, 128)
        self.bn_ip2 = nn.BatchNorm1d(128)
        self.ip3 = nn.Linear(128, 42)
        # cls branch
        self.conv4_2_cls = nn.Conv2d(40, 40, 3, 1, 1)
        self.bn4_2_cls = nn.BatchNorm2d(40)
        self.ip1_cls = nn.Linear(4 * 4 * 40, 128)
        self.bn_ip1_cls = nn.BatchNorm1d(128)
        self.ip2_cls = nn.Linear(128, 128)
        self.bn_ip2_cls = nn.BatchNorm1d(128)
        self.ip3_cls = nn.Linear(128, 2)
        # common used
        self.prelu = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape: ', x.shape)
        x = self.ave_pool(self.prelu(self.conv1_1(x)))
        # print('x after block1 and pool shape should be 32x8x27x27: ', x.shape)     # good
        # block 2
        x = self.prelu(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu shape should be 32x16x25x25: ', x.shape) # good
        x = self.prelu(self.conv2_2(x))
        # print('b2: after conv2_2 and prelu shape should be 32x16x23x23: ', x.shape) # good
        x = self.ave_pool(x)
        # print('x after block2 and pool shape should be 32x16x12x12: ', x.shape)
        # block 3
        x = self.prelu(self.conv3_1(x))
        # print('b3: after conv3_1 and pool shape should be 32x24x10x10: ', x.shape)
        x = self.prelu(self.conv3_2(x))
        # print('b3: after conv3_2 and pool shape should be 32x24x8x8: ', x.shape)
        x = self.ave_pool(x)
        # print('x after block3 and pool shape should be 32x24x4x4: ', x.shape)
        # block 4
        x = self.prelu(self.conv4_1(x))
        # print('x after conv4_1 and pool shape should be 32x40x4x4: ', x.shape)

        # points branch
        ip3 = self.prelu(self.conv4_2(x))
        # print('pts: ip3 after conv4_2 and pool shape should be 32x80x4x4: ', ip3.shape)
        ip3 = ip3.view(-1, 4 * 4 * 80)
        # print('ip3 flatten shape should be 32x1280: ', ip3.shape)
        ip3 = self.prelu(self.ip1(ip3))
        # print('ip3 after ip1 shape should be 32x128: ', ip3.shape)
        ip3 = self.prelu(self.ip2(ip3))
        # print('ip3 after ip2 shape should be 32x128: ', ip3.shape)
        ip3 = self.ip3(ip3)
        # print('ip3 after ip3 shape should be 32x42: ', ip3.shape)

        # cls branch
        ip3_cls = self.prelu(self.conv4_2_cls(x))
        # print('cls: ip3_cls after conv4_2 and pool shape should be 32x40x4x4: ', ip3_cls.shape)
        ip3_cls = ip3_cls.view(-1, 4 * 4 * 40)
        # print('cls: ip3_cls flatten shape should be 32x640: ', ip3_cls.shape)
        ip3_cls = self.prelu(self.ip1_cls(ip3_cls))
        ip3_cls = self.prelu(self.ip2_cls(ip3_cls))
        ip3_cls = self.ip3_cls(ip3_cls)
        # print('ip3_cls shape: ', ip3_cls.shape)

        return ip3, ip3_cls