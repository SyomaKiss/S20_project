from torch.optim.lr_scheduler import *


def get_scheduler(optimizer, entry):
    """
    :param optimizer: Wrapped optimizer
    :param entry: experiment configuration file
    :return: scheduler
    """
    entry = entry[0]
    kwargs = {key: entry[key] for key in entry.keys() if key not in ["name"]}
    ret = globals()[entry["name"]](optimizer, **kwargs)
    return ret