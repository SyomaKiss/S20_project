import importlib
import os
import sys
from os.path import exists, join

import torch

from src.configs import config as configs


def load_config_dump(name):
    config = configs.get_default()
    if "/" not in name:
        if exists(config.system.checkpoints_root) and \
                name in os.listdir(config.system.checkpoints_root) and \
                exists(join(config.system.checkpoints_root, name, "{}.yaml".format(name))):
            save_dir = join(config.system.checkpoints_root, name)
            filepath = join(save_dir, "{}.yaml".format(name))
        elif exists('configs') and '{}.yaml'.format(name) in os.listdir('configs'):
            filepath = join('configs', "{}.yaml".format(name))
            dump_config(configs.get_configuration(filepath))
        else:
            print("Unable to find config file \'{}\' either in \'{}\' or \'{}\'".format(name,
                                                                                        config.system.checkpoints_root,
                                                                                        'configs'), file=sys.stderr)
            raise FileNotFoundError
    else:
        filepath = name
    print(filepath)
    return configs.get_configuration(filepath)


def dump_config(config):
    save_dir = join(config.system.checkpoints_root, config.name)
    if not exists(save_dir):
        os.makedirs(save_dir)
    filepath = join(save_dir, "{}.yaml".format(config.name))
    with open(filepath, "w") as f:
        print(config, file=f)


def dump_model(config, model, tag=None):
    save_dir = join(config.system.checkpoints_root, config.name)
    if not exists(save_dir):
        os.makedirs(save_dir)

    if isinstance(model, dict):
        for key, value in model.items():
            if tag:
                torch.save(value.state_dict(), join(save_dir, "{}_{}.pth".format(key, tag)))
            torch.save(value.state_dict(), join(save_dir, "{}_{}.pth".format(key, 'latest')))
    elif isinstance(model, torch.nn.Module):
        if tag:
            torch.save(model.state_dict(), join(save_dir, "{}.pth".format(tag)))
        torch.save(model.state_dict(), join(save_dir, "{}.pth".format('latest')))
    else:
        print("Cant save object with type {}".format(type(model)))
        raise NotImplementedError('Error')


def get_logger(config):
    """
    Setup an output stream for experiments and init metrics logger
    :param config: experiment configuration file
    :return: metric summarizer, which is toTensorBoardWriter
    """

    save_dir = join(config.system.checkpoints_root, config.name)
    if not exists(save_dir):
        os.makedirs(save_dir)
        
    if config.system.stream == "file":
        sys.stdout = open(join(config.system.checkpoints_root, config.name, 'logs.txt'), 'a+')
        sys.stderr = sys.stdout

    logger_module = importlib.import_module('protocols.{}.logger'.format(config.protocol))
    return logger_module.MetricsLogger(config)


def get_latest_epoch_num(config):
    l = os.listdir(join(config.system.checkpoints_root, config.name))
    l = [int(i[2:-4]) for i in l if i.endswith(".pth") and i[2:-4].isdigit()]
    if l:
        return int(max(l))
    else:
        return 0

