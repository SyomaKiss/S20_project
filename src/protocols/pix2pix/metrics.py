from pytorch_msssim import SSIM as __SSIM__
from torch import nn
from torch.nn import *


def get_metrics(config):
    metrics_list = config.testing.metrics
    funcs = {}
    for entry in metrics_list:
        funcs[entry['name']] = __get_metric__(entry['name'])
    return funcs


def __get_metric__(name):
    if 'get_' + str(name) in globals():
        ret = globals()['get_' + name]()
    elif name in globals():
        ret = globals()[name]()
    else:
        raise NotImplementedError
    return ret


class MAE(nn.Module):
    """
    Inputs: shape (batch_size, #_of_points,  #_of_axes)
            [x,y]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x['x']
        f = L1Loss(reduction='none')
        return f(x, y).mean([-1, -2, -3])


class SSIM(__SSIM__):

    def __init__(self, data_range=1., **kwargs):
        super().__init__(data_range=float(data_range), **kwargs)

    def forward(self, x, y):
        x = x['x']
        return super().forward(x, y)
