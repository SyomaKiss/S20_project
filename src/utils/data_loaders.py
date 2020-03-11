import importlib

from torch.utils.data import DataLoader


def get_DataLoader(config, phase='train'):
    datasets_module = importlib.import_module('protocols.{}.dataset'.format(config.protocol))
    batch_size = config.dataset[phase].batch_size
    dataset = datasets_module.CustomDataset(config, phase=phase)
    shuffle = config.dataset[phase].shuffle
    num_workers = config.system.num_workers
    return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)