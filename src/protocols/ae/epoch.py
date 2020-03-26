import importlib
from pprint import pprint
from time import time

import torch

from utils import saver


def train_one_epoch(
        config,
        model,
        device,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        epoch,
        logger,
        log_interval=20,
):
    # dynamic import
    datasets = importlib.import_module('protocols.{}.dataset'.format(config.protocol))

    logger.reset()
    model.train()
    train_loss = 0
    start_time = time()

    # start of epoch ----------------------------------------------------------------
    for repetition_number in range(int(config.training.repeat_dataset) + 1):
        for batch_idx, batch_data in enumerate(train_loader):
            data, target = batch_data["image"], batch_data["target"]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, bbox=batch_data["bbox"])
            loss.backward()
            optimizer.step()

            logger.add(target, output.detach())

            if batch_idx % log_interval == 0:
                if config.training.save_visuals:
                    datasets.save_inference(
                        config,
                        "train",
                        batch_data,
                        output,
                        'ep-{}_b-{}_rep-{}'.format(epoch, batch_idx, repetition_number),
                    )

            train_loss += loss.item()  # sum up batch loss
    # end of epoch ----------------------------------------------------------------
    train_loss /= len(train_loader) * (config.training.repeat_dataset + 1)

    if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()

    # save model state
    if epoch % config.training.dump_period == 0:
        saver.dump_model(config, model, tag=epoch)

    res = logger.calculate(phase='train')
    print(
        "Train Epoch: {}/{} finished in {} min. {} sec. \tLoss: {:.6f}".format(
            epoch,
            config.training.num_epochs + config.model.load_state,
            int((time() - start_time) // 60),
            int((time() - start_time) % 60),
            train_loss,
        )
    )
    res["loss"] = train_loss
    res["lr"] = scheduler.get_lr()[0]
    pprint(res)
    return res


def test(
        config,
        model,
        device,
        test_loader,
        criterion,
        logger,
        phase="val",
        tag="0",
        log_interval=8,
):
    # dynamic import
    datasets = importlib.import_module('protocols.{}.dataset'.format(config.protocol))

    logger.reset()
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            data, target = (
                batch_data["image"],
                batch_data["target"]
            )
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, bbox=batch_data["bbox"]).item()

            logger.add(target, output.detach(), meta={'img_name': batch_data["img_name"]})

            if config.testing.save_visuals and batch_idx % log_interval == 0:
                datasets.save_inference(
                    config, phase, batch_data, output,
                    'ep-{}_b-{}'.format(tag, batch_idx),
                )

    test_loss /= len(test_loader)

    print(
        "{} set after {}: Average loss: {:.4f}".format(
            phase.capitalize(), tag, test_loss
        )
    )

    res = logger.calculate(phase=phase)
    res["loss"] = test_loss
    pprint(res)
    return res
