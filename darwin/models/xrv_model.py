from numpy import block
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchxrayvision import models
class DenseNetXRVFeature(nn.Module):
    """
    Modified Densenet-BC feature extractor.
    based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Modified from torchxrayvision's DenseNet class
    Args:
        pretrain_weights (string) = torchxrayvision pretrained model name
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    Note:
        Code adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py
    """

    def __init__(self, 
                 pretrain_weights=None,
                 growth_rate=32, 
                 block_config=(6, 12, 24, 16), 
                 num_init_features=64, 
                 bn_size=4,
                 drop_rate=0.5,
                 in_channels=1,
                 ):
        super(DenseNetXRVFeature, self).__init__()
        self.model_xrv_densenet = models.DenseNet(weights=pretrain_weights,
                                         growth_rate=growth_rate,
                                         block_config=block_config,
                                         num_init_features=num_init_features,
                                         bn_size=bn_size,
                                         drop_rate=drop_rate,
                                         in_channels=in_channels)
        self.features = self.model_xrv_densenet.features
        self.upsample = self.model_xrv_densenet.upsample
        self._out_features = self.model_xrv_densenet.classifier.in_features
        self.fc1 = nn.Linear(1024,15)
    def forward(self, x):
        x = models.fix_resolution(x, 224, self)
        models.warn_normalization(x)
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        
        out = self.fc1(out)
        return out

    def output_size(self):
        return self._out_features