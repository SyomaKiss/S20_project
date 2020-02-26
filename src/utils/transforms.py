from albumentations.core.composition import *
from albumentations.augmentations.transforms import *
from albumentations.pytorch import ToTensor, ToTensorV2


def parse_transforms(config):
    augmentations = []
    for aug in config:
        if aug['name'] in ['Compose', 'OneOf', 'OneOrOther']:
            tmp_augs = parse_transforms(aug['augmentations'])  # get inner list of augmentations
            params = {key: aug[key] for key in aug.keys() if key not in ['name', 'augmentations']}
            transform_class = globals()[aug['name']](tmp_augs, **params)
            augmentations.append(transform_class)
        else:
            params = {key: aug[key] for key in aug.keys() if key not in ['name', 'augmentations']}
            transform_class = globals()[aug['name']](**params)
            augmentations.append(transform_class)

    return augmentations


def get_augmentations(config):
    return Compose(parse_transforms(config.dataset.augmentations), additional_targets={'target': 'image'},
                   keypoint_params=KeypointParams(format='xy', remove_invisible=False))


def get_transforms(config):
    return Compose(parse_transforms(config.dataset.transforms), additional_targets={'target': 'image'},
                   keypoint_params={'format': 'xy', "remove_invisible": False})
