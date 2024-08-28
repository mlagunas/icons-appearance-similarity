from functools import reduce

import torch.nn as nn


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


model = nn.Sequential(  # Sequential,
    nn.Conv2d(1, 32, (5, 5), (1, 1), (1, 1)),
    nn.BatchNorm2d(32),
    nn.MaxPool2d((2, 2), (2, 2)),
    nn.Conv2d(32, 64, (5, 5), (1, 1), (1, 1)),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((2, 2), (2, 2)),
    nn.Conv2d(64, 64, (5, 5), (1, 1), (1, 1)),
    nn.BatchNorm2d(64),
    nn.MaxPool2d((2, 2), (2, 2)),
    nn.Conv2d(64, 128, (5, 5), (1, 1), (1, 1)),
    nn.BatchNorm2d(128),
    nn.MaxPool2d((2, 2), (2, 2)),
    Lambda(lambda x: x.view(x.size(0), -1)),  # View,
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(10368, 4096)),  # Linear,
    nn.Dropout(0.3),
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 1024)),  # Linear,
    nn.Dropout(0.3),
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(1024, 256)),  # Linear,
)
