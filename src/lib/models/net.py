import os
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn.tasks import DetectionModel


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model, pretrained=''):
    for i, m in enumerate(model.modules()):
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # 原地操作

    for head in model.heads:
        final_layer = model.__getattr__(head)
        for i, m in enumerate(final_layer.modules()):
            if isinstance(m, nn.Conv2d):
                if m.weight.shape[0] == model.heads[head]:
                    if 'hm' in head:
                        nn.init.constant_(m.bias, -2.19)
                    else:
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)

    if os.path.isfile(pretrained):
        ckpt = torch.load(pretrained)  # load checkpoint
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.backbone.state_dict())  # intersect
        model.backbone.load_state_dict(state_dict, strict=False)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), pretrained))  # report


class Net(nn.Module):
    def __init__(self, heads, config_file):
        self.heads = heads
        super(Net, self).__init__()
        self.backbone = DetectionModel(config_file)
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(64, num_output, kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

    def forward(self, x):
        x = self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        if self.training:
            return [ret]
        else:
            # 方便onnx导出
            hm = ret["hm"]
            wh = ret["wh"]
            reg = ret["reg"]
            hm = F.sigmoid(hm)
            hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)

            id_feature = ret['id']
            id_feature = F.normalize(id_feature, dim=1)
            id_feature = id_feature.permute(0, 2, 3, 1).contiguous()  # switch id dim
            wh = wh.permute(0, 2, 3, 1).contiguous()  # switch id dim
            reg = reg.permute(0, 2, 3, 1).contiguous()  # switch id dim

            return [hm, wh, reg, hm_pool, id_feature]


def get__net(num_layers, heads, head_conv):
    config_file = os.path.join(
        os.path.dirname(__file__),
        'networks/config/yolov8s.yaml'
    )
    pretrained = os.path.join(
        os.path.dirname(__file__),
        '../../../models/yolov8s.pt'
    )

    model = Net(heads, config_file)
    initialize_weights(model, pretrained)
    return model
