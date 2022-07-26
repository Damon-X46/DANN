import os, sys

base_path = os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
import torch.nn as nn
from models.functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()      # 假设输入的是[128, 3, 64, 40]
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 128, kernel_size=3))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(128))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_conv3', nn.Conv2d(128, 256, kernel_size=3))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(256))
        self.feature.add_module('f_relu3', nn.ReLU(True))
        self.feature.add_module('f_pool3', nn.MaxPool2d(2))


        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(256 * 6 * 3, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256 * 6 * 3, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):       # input_data[128, 3, 64, 40]
        input_data = input_data.expand(input_data.data.shape[0], 3, 64, 40)     # [128, 3, 64, 40]
        feature = self.feature(input_data)      # [128, 256, 6, 3]
        feature = feature.view(-1, 256 * 6 * 3)  # [128, 4608]
        reverse_feature = ReverseLayerF.apply(feature, alpha)                   # [128, 4608]
        class_output = self.class_classifier(feature)                           # [128, 10]         # 用来分是哪个数字(0-9)
        domain_output = self.domain_classifier(reverse_feature)                 # [128, 2]          # 用来分是哪个域(源域还是目标域)

        return class_output, domain_output


if __name__ == '__main__':
    model = CNNModel()
    batches = torch.normal(0, 1, (128, 3, 64, 40))
    pre = model(batches, 0.0)
    pass