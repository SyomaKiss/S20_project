"""
Dataset class must be called "CustomDataset"
"""

import os
from os.path import join, exists

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

import utils
from utils.transforms import get_augmentations, get_transforms


class CustomDataset(Dataset):

    def __init__(self, config, phase='train'):
        self.df = pd.read_csv(config.dataset[phase].csv_path)
        self.phase = phase
        self.transforms = get_transforms(config)
        self.augmentations = get_augmentations(config) if 'train' in phase else None
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: {..., bbox(4, 2): [upLeft, upRight, downLeft, downRight]}
        """
        img_name = self.df.loc[idx, 'Image Index']
        img_path = os.path.join(self.config.dataset.root,
                                img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = np.array(eval(self.df.loc[idx, 'bbox']))

        resize_pair = albu.Compose([albu.Resize(self.config.dataset.img_size, self.config.dataset.img_size)],
                                   keypoint_params={'format': 'xy', "remove_invisible": False})
        augmented = resize_pair(image=image, keypoints=bbox)

        image = augmented['image']
        bbox = np.round(np.array(augmented['keypoints'])).astype(int)

        target = np.copy(image)
        target[bbox[0, 1]:bbox[2, 1], bbox[0, 0]:bbox[1, 0], :] = 0

        sample = {'image': target, 'target': image, 'bbox': bbox, 'img_name': img_name}

        if self.augmentations and 'train' in self.phase:
            augmented = self.augmentations(image=sample['image'], target=sample['target'], keypoints=sample['bbox'])
            if ((np.array(augmented['keypoints']).min() < 0) or
                    (np.array(augmented['keypoints']).max() > self.config.dataset.img_size)):
                print('Wrong augmentations')
            else:
                sample['image'] = augmented['image']
                sample['target'] = augmented['target']
                sample['bbox'] = np.array(augmented['keypoints'])

        if self.transforms:
            # Apply transform to numpy.ndarray which represents sample image
            transformed = self.transforms(image=sample['image'], target=sample['target'], keypoints=sample['bbox'])
            sample['image'] = transformed['image']
            sample['target'] = transformed['target']
            sample['bbox'] = torch.tensor(transformed['keypoints'])

        sample['image'] = sample['image'].float()
        sample['target'] = sample['target'].float()
        sample['bbox'] = sample['bbox'].float()
        return sample

    @staticmethod
    def show(image=None, target=None, output=None, landmarks=None, plot_point_width=3, min_px_value=0, max_px_value=1):
        """
        All variables must be of the same type, ndarray or torch.Tensor
        :param max_px_value:
        :param min_px_value:
        :param image: (H, W, CH)
        :param target: (H, W, CH)
        :param output: (H, W, CH)
        :param landmarks: bbox (4, 2) [upLeft, upRight, downLeft, downRight]
        :param plot_point_width: scatter point width
        :return: scat
        """

        empty_shape = image.shape if image is not None else target.shape if target is not None \
            else output if output is not None else (512, 512, 3)
        if isinstance(output, torch.Tensor) or isinstance(target, torch.Tensor) or isinstance(image, torch.Tensor) or \
                isinstance(landmarks, torch.Tensor):
            new_axes = [0, 2, 3, 1] if len(empty_shape) == 4 else [1, 2, 0]
            if output is not None: output = output.cpu().detach().numpy().transpose(*new_axes)
            if target is not None: target = target.cpu().detach().numpy().transpose(*new_axes)
            if image is not None: image = image.cpu().detach().numpy().transpose(*new_axes)
            if landmarks is not None: landmarks = landmarks.cpu().detach().numpy()

        empty_shape = image.shape if image is not None else target.shape if target is not None \
            else output if output is not None else (512, 512, 3)

        if len(empty_shape) == 4:
            empty_shape = empty_shape[1:]
            n_in_row = 2
            rows = np.ceil(len(image) / n_in_row).astype(int)

            fig, ax = plt.subplots(rows, 3 * n_in_row, sharey=True, figsize=(3 * n_in_row * 10, rows * 10))
            for i, _ in enumerate(image):
                offset = i % 2 * 3

                ax[i // 2 - i % 2, 0 + offset].imshow(np.zeros(empty_shape)) if image is None else ax[
                    i // 2 - i % 2, 0 + offset].imshow(image[i], vmin=min_px_value, vmax=max_px_value)
                ax[i // 2 - i % 2, 1 + offset].imshow(np.zeros(empty_shape)) if target is None else ax[
                    i // 2 - i % 2, 1 + offset].imshow(target[i], vmin=min_px_value, vmax=max_px_value)
                ax[i // 2 - i % 2, 2 + offset].imshow(np.zeros(empty_shape)) if output is None else ax[
                    i // 2 - i % 2, 2 + offset].imshow(output[i], vmin=min_px_value, vmax=max_px_value)

                if landmarks is not None:
                    ax[i // 2 - i % 2, 0 + offset].scatter(landmarks[i, :, 0], landmarks[i, :, 1],
                                                           c='r', s=2 ** plot_point_width)
                    ax[i // 2 - i % 2, 1 + offset].scatter(landmarks[i, :, 0], landmarks[i, :, 1],
                                                           c='r', s=2 ** plot_point_width)
                    ax[i // 2 - i % 2, 2 + offset].scatter(landmarks[i, :, 0], landmarks[i, :, 1],
                                                           c='r', s=2 ** plot_point_width)
        else:

            fig, ax = plt.subplots(1, 3, sharey=True, figsize=(30, 10))

            ax[0].imshow(np.zeros(empty_shape)) if image is None else ax[0].imshow(image,
                                                                                   vmin=min_px_value,
                                                                                   vmax=max_px_value)
            ax[1].imshow(np.zeros(empty_shape)) if target is None else ax[1].imshow(target,
                                                                                    vmin=min_px_value,
                                                                                    vmax=max_px_value)
            ax[2].imshow(np.zeros(empty_shape)) if output is None else ax[2].imshow(output,
                                                                                    vmin=min_px_value,
                                                                                    vmax=max_px_value)

            if landmarks is not None:
                ax[0].scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=2 ** plot_point_width)
                ax[1].scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=2 ** plot_point_width)
                ax[2].scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=2 ** plot_point_width)
        fig.savefig('tmp.png')
        return fig


def save_inference(config, phase, input_d, output, tag='default_tag'):
    """Show image with landmarks for a batch of samples."""
    min_px_value, max_px_value = utils.PIXEL_RANGES[config.model.final_activation[0]['name']]

    image, target, landmarks = input_d['image'], input_d['target'], input_d['bbox']
    fig = CustomDataset.show(image, target, output, landmarks, min_px_value=min_px_value, max_px_value=max_px_value)
    save_dir = join(config.system.checkpoints_root, config.name, 'visuals', phase)
    if not exists(save_dir): os.makedirs(save_dir)
    fig.savefig(join(save_dir, tag + '.png'))
    plt.close(fig)
