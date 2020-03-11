import numpy as np
import torch
from torch.nn import *


def get_loss(entry):
    ret = {}
    for entry in entry:
        kwargs = {key: entry[key] for key in entry.keys() if key not in ['name']}
        if 'get_' + str(entry['name']) in globals():
            ret[entry['name']] = globals()['get_' + str(entry['name'])](entry, **kwargs)
        elif entry['name'] in globals():
            ret[entry['name']] = globals()[entry['name']](**kwargs)
        else:
            raise NotImplementedError

    return ret


class MAE_bbox_weighted(Module):

    def __init__(self, alpha=100, border_weighting=None, margin=5, border_constant=None):
        """
        Gives a privilege to an area inside a bbox
        :param alpha: weight assigned to pixels inside bbox
        :param border_weighting: weighting policy applied to bbox neighbouring pixels [None, 'constant', 'linear']
        :param margin: width of a neighbour area
        :param border_constant: used with 'constant' border_weighting policy, weight assigned to neighbouring pixels
        """
        super().__init__()
        self.alpha = alpha
        self.border_weighting = border_weighting
        self.border_constant = self.alpha / 2 if border_constant is None else border_constant
        self.margin = margin
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, y, bbox=None):
        """
        :param x: batch of predictions
        :param y: batch of targets
        :param bbox: (batch_size, 4, 2): [upLeft, upRight, downLeft, downRight]
        :return: mean of weighted errors across arrays
        """
        bbox = bbox.int()
        margin = self.margin

        ret = abs(x - y)

        for i, value in enumerate(ret):
            y_min, y_max = bbox[i][:, 1].min(), bbox[i][:, 1].max()
            x_min, x_max = bbox[i][:, 0].min(), bbox[i][:, 0].max()
            ret[i][:, y_min:y_max, x_min:x_max] *= self.alpha
            if x_min - margin > 0 and x_max + margin < ret[i].shape[-1] and y_min - margin > 0 and y_max + margin < \
                    ret[i].shape[-2]:

                if self.border_weighting == 'constant':
                    ret[i][:, y_min - margin:y_min, x_min - margin:x_max + margin] *= self.border_constant
                    ret[i][:, y_max: y_max + margin, x_min - margin:x_max + margin] *= self.border_constant
                    ret[i][:, y_min:y_max, x_min - margin:x_min] *= self.border_constant
                    ret[i][:, y_min:y_max, x_max:x_max + margin] *= self.border_constant
                if self.border_weighting == 'linear':
                    margin = self.margin

                    multiplier = torch.tensor(np.arange(1, margin + 1)[None, :, None], dtype=ret.dtype,
                                                                                        device=self.device)
                    multiplier = multiplier / margin * self.alpha
                    ret[i][:, y_min - margin:y_min, x_min:x_max] *= multiplier

                    multiplier = torch.tensor(np.arange(margin, 0, -1)[None, :, None], dtype=ret.dtype,
                                                                                        device=self.device)
                    multiplier = multiplier / margin * self.alpha
                    ret[i][:, y_max: y_max + margin, x_min:x_max] *= multiplier

                    multiplier = torch.tensor(np.arange(1, margin + 1)[None, None, :], dtype=ret.dtype,
                                                                                        device=self.device)
                    multiplier = multiplier / margin * self.alpha
                    ret[i][:, y_min:y_max, x_min - margin:x_min] *= multiplier

                    multiplier = torch.tensor(np.arange(margin, 0, -1)[None, None, :], dtype=ret.dtype,
                                                                                        device=self.device)
                    multiplier = multiplier / margin * self.alpha
                    ret[i][:, y_min:y_max, x_max:x_max + margin] *= multiplier

                    # create corner matrix
                    a = np.arange(1, margin + 1)[None, :]
                    a = np.minimum(a, a.T)[None, :, :]
                    # up left corner
                    multiplier = torch.tensor(a / margin * self.alpha, dtype=ret.dtype, device=self.device)
                    ret[i][:, y_min - margin:y_min, x_min - margin:x_min] *= multiplier
                    # down left corner
                    a = np.rot90(a, k=1, axes=(1, 2))
                    multiplier = torch.tensor(a / margin * self.alpha, dtype=ret.dtype, device=self.device)
                    ret[i][:, y_max:y_max + margin, x_min - margin:x_min] *= multiplier
                    # down right corner
                    a = np.rot90(a, k=1, axes=(1, 2))
                    multiplier = torch.tensor(a / margin * self.alpha, dtype=ret.dtype, device=self.device)
                    ret[i][:, y_max:y_max + margin, x_max:x_max + margin] *= multiplier
                    # up right corner
                    a = np.rot90(a, k=1, axes=(1, 2))
                    multiplier = torch.tensor(a / margin * self.alpha, dtype=ret.dtype, device=self.device)
                    ret[i][:, y_min - margin:y_min, x_max:x_max + margin] *= multiplier

        return ret.mean()
