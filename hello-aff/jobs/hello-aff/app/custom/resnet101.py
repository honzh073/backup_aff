# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# from torchvision.models import resnet101, ResNet101_Weights

# class resnet101(nn.Module):
#     torch.hub.set_dir('/local/data1/honzh073/download/TORCH_PRETRAINED')

#     def __init__(self, num_classes=2):
#         super(resnet101, self).__init__()
#         self.resnet101 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
#         for param in self.resnet101.parameters():
#             param.requires_grad = True
#         self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.resnet101(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet101(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet101, self).__init__()
        self.inplanes = 64  # 初始的inplanes值
        # 定义ResNet-101的前几层，不包括最后的全连接层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 23, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类层
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride))
        self.inplanes = planes * 4  # Bottleneck的输出通道数是planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将特征图展平
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
