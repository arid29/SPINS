# Adapted LPIPS implementation; upstream notice: licenses/LPIPS-LICENSE.
# modified to load weights from local pretrained/ directory (no network access needed)
import os
import torch
import torch.nn as nn
import torchvision
from collections import namedtuple

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained')


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # load VGG16 features from local weights
        if pretrained:
            model = torchvision.models.vgg16(weights=None)
            local_path = os.path.join(PRETRAINED_DIR, 'vgg16_features.pth')
            model.features.load_state_dict(torch.load(local_path, map_location='cpu', weights_only=True))
            vgg_pretrained_features = model.features
        else:
            vgg_pretrained_features = torchvision.models.vgg16(weights=None).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS(nn.Module):
    def __init__(self, net='vgg', pretrained=True, use_dropout=True):
        super().__init__()
        self.net_type = net

        if self.net_type in ['vgg', 'vgg16']:
            self.chns = [64, 128, 256, 512, 512]
            self.net = vgg16(pretrained=pretrained, requires_grad=False)
            self.lpips_weights = os.path.join(PRETRAINED_DIR, 'lpips_vgg.pth')
        else:
            raise NotImplementedError(f"Only 'vgg' net is supported with local weights, got '{net}'")

        self.scaling_layer = ScalingLayer()
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])

        if pretrained:
            self.load_state_dict(torch.load(self.lpips_weights, map_location='cpu', weights_only=True), strict=False)

        self.eval()

    def forward(self, input, target):
        input, target = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(input), self.net(target)

        val = 0
        for i in range(len(self.chns)):
            feats0, feats1 = normalize_tensor(outs0[i]), normalize_tensor(outs1[i])
            diffs = (feats0 - feats1) ** 2
            val += spatial_average(self.lins[i].model(diffs), keepdim=True)

        return val
