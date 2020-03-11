import os
from os.path import join, exists

from torch.utils.tensorboard import SummaryWriter


class CustomWriter:
    def __init__(self, config):
        """
        Custom implementation of Tensor Board writer
        :param config: experiment configuration file
        """

        save_dir = join(config.system.checkpoints_root, "tensorboard_logs", str(config.name))
        if not exists(save_dir):
            os.makedirs(save_dir)

        self.config = config
        self.writer = SummaryWriter(save_dir)

    def write_result(self, result_dict, epoch, phase):
        for name, value in result_dict.items():
            self.write_single(str(phase + "_" + name), value, epoch)

    def write_single(self, name, value, epoch):
        self.writer.add_scalar(self.config.name + "/" + name, value, epoch)

    def flush(self):
        self.writer.flush()
