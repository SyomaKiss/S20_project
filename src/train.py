import importlib
import random
import sys

import numpy as np
import torch
# from tqdm import tqdm

from models import pool as models
from utils import optimizers, schedulers, losses
from utils import saver
from utils.tb_writers import CustomWriter
from utils.data_loaders import get_DataLoader


def set_seed(config):
    seed = config.system.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(config):
    tb_writer = CustomWriter(config)
    logger = saver.get_logger(config)

    num_epochs = config.training.num_epochs
    model = {'G': models.get_model(config, tag='G'), 'D': models.get_model(config, tag='D')}
    model = {key: torch.nn.DataParallel(value) for key, value in model.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_DataLoader(config, phase="train")
    val_loader = get_DataLoader(config, phase="val")

    optimizer = {'G': optimizers.get_optimizer(model['G'].parameters(), config.model.G.optimizer),
                 'D': optimizers.get_optimizer(model['D'].parameters(), config.model.D.optimizer)}

    scheduler = {'G': schedulers.get_scheduler(optimizer['G'], config.model.G.scheduler),
                 'D': schedulers.get_scheduler(optimizer['D'], config.model.D.scheduler)}

    criterion = {'G': losses.get_loss(config.model.G.criterion),
                 'D': losses.get_loss(config.model.D.criterion)}

    start_epoch_num = (
        saver.get_latest_epoch_num(config)
        if config.model.G.load_state == -1 or config.model.G.load_state == "latest"
        else config.model.G.load_state
    )

    # Dynamic imports according to protocols
    epoch_module = importlib.import_module('protocols.{}.epoch'.format(config.protocol))
    train_one_epoch, test = getattr(epoch_module, 'train_one_epoch'), getattr(epoch_module, 'test')

    for epoch in range(start_epoch_num + 1, start_epoch_num + num_epochs + 1):

        train_buffer = train_one_epoch(
            config=config,
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=epoch,
            logger=logger,
            log_interval=config.training.log_interval,
        )
        tb_writer.write_result(train_buffer, epoch, phase="train")

        if epoch % config.training.validation_period == 0:
            val_buffer = test(
                config=config,
                model=model,
                device=device,
                test_loader=val_loader,
                criterion=criterion,
                logger=logger,
                phase="val",
                tag=epoch,
                log_interval=8,
            )
            tb_writer.write_result(val_buffer, epoch, phase="val")


if __name__ == "__main__":
    config = saver.load_config_dump(sys.argv[1])

    set_seed(config)
    run_training(config)
