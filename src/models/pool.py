import os

import torch
import importlib

from torch import nn

from utils.activations import get_activation
from utils.param_initialization import get_init_func


def get_model(config, tag='G'):
    """
    1) define arch.
    2) append final act.
    3) a) load checkpoint
        b) init params.
    4) .to(dev.)

    :param config:
    :param tag: title of model in CONFIG
    :return:
    """
    models_module = importlib.import_module('protocols.{}.models'.format(config.protocol))
    model_config = getattr(config.model, tag)

    model = getattr(models_module, model_config.name)(**model_config.kwargs)

    model = nn.Sequential(model, get_activation(entry=model_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_config.load_state == 0:
        def weights_init(m):
            func = get_init_func(model_config)
            if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
                func(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        print('Using initialization function')
        model.apply(weights_init)
    elif model_config.load_state == -1 or model_config.load_state == "latest":
        path_to_weights = os.path.join(config.system.checkpoints_root, config.name, '{}_latest.pth'.format(tag))
        state_dict = torch.load(path_to_weights)
        if "module" in list(state_dict.keys())[0]:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print('Restore the model from: {}'.format(path_to_weights))
    else:
        try:
            path_to_weights = model_config.load_state
            state_dict = torch.load(path_to_weights)
            if "module" in list(state_dict.keys())[0]:
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict

            model.load_state_dict(state_dict)
            print('Restore the model from: {}'.format(path_to_weights))
        except FileNotFoundError:
            print('Unable to load model weights. Please check "config.model.load_state" field. It must be either '
                  'checkpoint number or path to existing weights dump')
    return model.to(device)
