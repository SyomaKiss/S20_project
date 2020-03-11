import importlib
from os.path import join

import pandas as pd
import torch


class MetricsLogger:
    def __init__(self, config):
        self.aggregated_true = []
        self.aggregated_pred = []
        self.meta = {}
        self.config = config
        metrics_module = importlib.import_module('protocols.{}.metrics'.format(config.protocol))
        self.funcs = metrics_module.get_metrics(config)
        self.img_size = 256

    def reset(self):
        self.__init__(self.config)

    def add(self, true, pred, **kwargs):
        assert isinstance(true, torch.Tensor), isinstance(pred, torch.Tensor)
        assert pred.shape == true.shape

        true = true.cpu().detach().double()
        pred = pred.cpu().detach().double()

        if 'img_size' in kwargs and kwargs['img_size'] != self.img_size:
            setattr(self, 'img_size', kwargs['img_size'])

        # stack metadata for all samples
        if 'meta' in kwargs:
            meta = kwargs['meta']
            if len(self.meta) == 0:
                self.meta = meta
            else:
                for key, value in meta.items():
                    if key not in self.meta:
                        continue
                    assert isinstance(self.meta[key], type(meta[key]))
                    if isinstance(meta[key], list):
                        self.meta[key] += meta[key]
                    elif isinstance(meta[key], torch.Tensor):
                        self.meta[key] = torch.cat([self.meta[key], meta[key]])
                    else:
                        print("Concatenation for {} not implemented".format(type(meta[key])))
                        raise NotImplementedError

        self.aggregated_true.append(true)
        self.aggregated_pred.append(pred)

    def calculate(self, phase='test'):
        self.aggregated_true = torch.cat(self.aggregated_true)
        self.aggregated_pred = torch.cat(self.aggregated_pred)

        x = {'x': self.aggregated_true, 'meta': self.meta, 'img_size': self.img_size}

        # compute metric for each sample
        ret = {
            key: func(x,
                      self.aggregated_pred
                      )
            for key, func in self.funcs.items()
        }
        for key, value in self.meta.items():    # ensure all meta data stored as lists
            if isinstance(value, torch.Tensor):
                self.meta[key] = value.tolist()
            assert isinstance(value, list), "Meta data must be stored as a list to be able to fit pd.Dataframe"

        if 'test' in phase:     # create .csv table with the results for each sample
            df = pd.DataFrame(self.meta)

            # add metrics to resulting table
            for key, value in ret.items():
                df[key] = value.tolist()

            for key, value in self.meta.items():
                df[key] = value

            # add header with  mean and std for each column
            df = pd.concat([df.describe(include='all').loc[['mean', 'std']], df])
            filepath = join(join(self.config.system.checkpoints_root, self.config.name), "{}.csv".format(self.config.name))
            df.to_csv(filepath, index=True)

        return {
            key: value.mean().item()
            for key, value in ret.items()
        }