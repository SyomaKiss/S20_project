from torch.nn import *


def get_activation(entry):
    entry = entry.final_activation[0]
    kwargs = {key: entry[key] for key in entry.keys() if key not in ['name']}
    if entry['name'] in globals():
        ret = globals()[entry['name']](**kwargs)
    else:
        raise NotImplementedError
    return ret

