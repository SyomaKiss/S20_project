import importlib
from pprint import pprint
from time import time

import numpy as np
import torch

from utils import saver, set_requires_grad


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

    net_G, net_D = model['G'], model['D']
    optimizer_G, optimizer_D = optimizer['G'], optimizer['D']
    scheduler_G, scheduler_D = scheduler['G'], scheduler['D']
    criterion_G, criterion_D = criterion['G'], criterion['D']

    criterionGAN = list(criterion_D.values())[0]

    net_G.train()
    net_D.train()

    train_loss = 0
    loss_D_fake = 0
    loss_D_real = 0
    loss_G_adversarial = 0
    start_time = time()

    # start of epoch ----------------------------------------------------------------
    for batch_idx, batch_data in enumerate(train_loader):

        data, target = batch_data["image"], batch_data["target"]
        data, target = data.to(device), target.to(device)
        output_G = net_G(data)

        # ------------------------Optimize D-------------------------
        set_requires_grad(net_D, True)  # enable backprop for D
        optimizer_D.zero_grad()

        # Fake discrimination
        input_D_fake = torch.cat((data, output_G), 1)
        output_D_fake = net_D(input_D_fake.detach())
        loss_D_fake = criterionGAN(output_D_fake, torch.tensor(0.).expand_as(output_D_fake).to(device))

        # Real discrimination
        input_D_real = torch.cat((data, target), 1)
        output_D_real = net_D(input_D_real)
        loss_D_real = criterionGAN(output_D_real, torch.tensor(1.).expand_as(output_D_real).to(device))

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizer_D.step()
        # ------------------------Optimize G-------------------------
        set_requires_grad(net_D, False)  # disable backprop for D
        optimizer_G.zero_grad()

        input_D_fake = torch.cat((data, output_G), 1)
        output_D_fake = net_D(input_D_fake.detach())
        loss_G_adversarial = criterionGAN(output_D_fake, torch.tensor(1.).expand_as(output_D_fake).to(device))

        loss_G_similarity = criterion_G['MAE_bbox_weighted'](output_G, target, bbox=batch_data["bbox"])
        # loss_G_SSIM = - np.log(criterion_G['SSIM'](output_G, target)) * 10000

        loss_G = loss_G_adversarial * config.coef.alphaGAN + loss_G_similarity * config.coef.alphaSIM

        loss_G.backward()
        optimizer_G.step()

        # loss = criterion(output, target, bbox=batch_data["bbox"])

        logger.add(target, output_G.detach())

        if batch_idx % log_interval == 0 and epoch % config.training.visualisation_period == 0:
            if config.training.save_visuals:
                datasets.save_inference(
                    config,
                    "train",
                    batch_data,
                    output_G,
                    'ep-{}_b-{}'.format(epoch, batch_idx),
                )

        train_loss += loss_G_similarity.item()  # sum up batch losses
    # end of epoch ----------------------------------------------------------------
    train_loss /= len(train_loader)
    loss_D_fake /= len(train_loader)
    loss_D_real /= len(train_loader)
    loss_G_adversarial /= len(train_loader)

    if isinstance(scheduler_D, torch.optim.lr_scheduler.StepLR):
        scheduler_D.step()
    if isinstance(scheduler_G, torch.optim.lr_scheduler.StepLR):
        scheduler_G.step()

    # save model state
    if epoch % config.training.dump_period == 0:
        saver.dump_model(config, model, tag=epoch)

    # print('{}m{}s'.format(int((time() - start_time) // 60), int((time() - start_time) % 60)))
    res = logger.calculate(phase='train')
    print(
        "Train Epoch: {}/{} finished in {} min. {} sec. \tLoss: {:.6f}".format(
            epoch,
            config.training.num_epochs + config.model.G.load_state,
            int((time() - start_time) // 60),
            int((time() - start_time) % 60),
            train_loss,
        )
    )
    res["loss_G_similarity"] = train_loss
    res["loss_D_fake"] = loss_D_fake
    res["loss_D_real"] = loss_D_real
    res["loss_G_adversarial"] = loss_G_adversarial
    res["lr"] = scheduler_G.get_lr()[0]
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

    net_G, net_D = model['G'], model['D']
    criterion_G, criterion_D = criterion['G'], criterion['D']

    net_G.eval()
    net_D.eval()

    logger.reset()
    test_loss = 0
    start_time = time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            data, target = batch_data["image"], batch_data["target"]
            data, target = data.to(device), target.to(device)

            output_G = net_G(data)

            test_loss += criterion_G['MAE_bbox_weighted'](output_G, target, bbox=batch_data["bbox"]).item()

            logger.add(target, output_G.detach(), meta={'img_name': batch_data["img_name"]})

            if config.testing.save_visuals and batch_idx % log_interval == 0:
                datasets.save_inference(
                    config, phase, batch_data, output_G,
                    'ep-{}_b-{}'.format(tag, batch_idx),
                )

    test_loss /= len(test_loader)
    print(
        "{} set after {}: Average loss: {:.4f}".format(
            phase.capitalize(), tag, test_loss
        )
    )
    # print('{}m{}s'.format(int((time() - start_time) // 60), int((time() - start_time) % 60)))
    res = logger.calculate(phase=phase)
    res["loss_G_similarity"] = test_loss
    pprint(res)
    return res
