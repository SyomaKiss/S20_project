from yacs.config import CfgNode as CN
import os

_C = CN()
_C.name = "default"
_C.protocol = "ae"


# System settings -----------------------------------------------
_C.system = CN()
_C.system.num_gpu = 1
_C.system.num_workers = 4
_C.system.checkpoints_root = "training_checkpoints"
_C.system.stream = 'file'
_C.system.seed = 777

# Dataset opt. -----------------------------------------------
_C.dataset = CN()
_C.dataset.repeat_dataset = 1
_C.dataset.cache = 1
_C.dataset.img_size = 512
_C.dataset.augmentations = []
_C.dataset.transforms = [{"name": "ToTensor"}]
_C.dataset.root = CN(new_allowed=True)
_C.dataset.root.NIH = "/datasets/ilyas/ChestNets/images"
_C.dataset.root.JSRT = "/home/semyon/cancer_astro/data/pngversion"
data_annotation_root = '/home/semyon/cancer_astra/data'

# Train
_C.dataset.train = CN()
_C.dataset.train.csv_path = os.path.join(data_annotation_root, "train.csv")
_C.dataset.train.batch_size = 16
_C.dataset.train.shuffle = True

# Val
_C.dataset.val = CN()
_C.dataset.val.csv_path = os.path.join(data_annotation_root, "val.csv")
_C.dataset.val.batch_size = 16
_C.dataset.val.shuffle = False


#Test
_C.dataset.test = CN()
_C.dataset.test.csv_path = os.path.join(data_annotation_root, "test.csv")
_C.dataset.test.batch_size = 16
_C.dataset.test.shuffle = False


# Model options -----------------------------------------------
_C.model = CN()
_C.model.G = CN()
_C.model.D = CN()

# G --------------------------------------
_C.model.G.name = 'Unet'
_C.model.G.in_channels = 3
_C.model.G.out_channels = 3
_C.model.G.final_activation = [{"name": "Sigmoid"}]
_C.model.G.params_init = "xavier_normal_"
_C.model.G.load_state = 0
_C.model.G.optimizer = [{"lr": 0.001, "name": "Adam"}]
_C.model.G.scheduler = [{"name": "StepLR", "step_size": 1000}]
_C.model.G.criterion = [{"name": "MSELoss"}]
_C.model.G.kwargs = CN(new_allowed=True)
_C.model.G.kwargs.input_nc = 3

# D ---------------------------------------
_C.model.D.name = 'NLayerDiscriminator'
_C.model.D.in_channels = 3
_C.model.D.out_channels = 3
_C.model.D.final_activation = [{"name": "Sigmoid"}]
_C.model.D.params_init = "xavier_normal_"
_C.model.D.load_state = 0
_C.model.D.optimizer = [{"lr": 0.001, "name": "Adam"}]
_C.model.D.scheduler = [{"name": "StepLR", "step_size": 1000}]
_C.model.D.criterion = [{"name": "MSELoss"}]
_C.model.D.kwargs = CN(new_allowed=True)
_C.model.D.kwargs.n_layers = 6

# Training options -----------------------------------------------
_C.training = CN()
_C.training.num_epochs = 2
_C.training.dump_period = 5  # period(# of epochs) of saving model state
_C.training.validation_period = 10  # period(# of epochs) per validation
_C.training.log_interval = 20  # period(# of batches) to save visuals/log info
_C.training.save_visuals = True
_C.training.visualisation_period = 50

_C.coef = CN(new_allowed=True)
_C.coef.alphaGAN = 1
_C.coef.alphaSIM = 10
_C.coef.bbox_margin = 10
# Testing options -----------------------------------------------
_C.testing = CN()
_C.testing.save_visuals = True

# only the metrics defined in utils.metrics can be used
_C.testing.metrics = [{"name": "MAE"}]


def get_default():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()


def get_configuration(filename):
    """
    Obtain dict-like configuration object based on default configuration and updated
    with specified file.
    :param filename: path to .yaml configs file
    :return: CfgNode object
    """
    cfg = get_default()
    cfg.merge_from_file(filename)


    # Uncomment not to be able modify values with dot notation
    # cfg.freeze()
    return cfg


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("default.yaml")
    print(_C)
    with open(sys.argv[1], "w") as f:
        print(_C, file=f)
