from torch.nn.init import *


def get_init_func(entry):
    entry = entry.params_init
    if entry in globals():
        ret = globals()[entry]
    else:
        raise NotImplementedError
    return ret

