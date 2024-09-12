import torch
from torch.utils.data import DataLoader
from torchinfo import summary as model_summary
import torchvision.transforms as Tr
import wandb
import copy
import collections
import logging, multiprocessing
import psutil
from gpiozero import CPUTemperature
import numpy as np
import sys, os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('../')

import config as cfg
from models.s3d import S3D
from models.hbco import HBCO
from models.tfe import TransformerEncoder_CLIENT as TFE_C
from models.tfe import TransformerEncoder_SERVER as TFE_S
from models.tfe import TransformerEncoder_UNIT as TFE_U
from dataset.dataloader import NEARMISS
from dataset.transformation.videotransforms import *
from dataset.transformation.my_transform import *
import dataset.transformation.my_transform as T

# Set random seeds for reproducibility
np.random.seed(cfg.random_seed)
torch.manual_seed(cfg.random_seed)


class Party:
    def __init__(self, name):
        super(Party, self).__init__()
        self.name = name

        # Check if GPU is available, otherwise use CPU
        if torch.cuda.is_available():
            logger.info(torch.cuda.get_device_name(0))
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Initialize dataloaders and datasets for train, val, test
        self.dataloaders = {'train': None, 'val': None, 'test': None}
        self.datasets = {'train': None, 'val': None, 'test': None}
        self._training_sizes = {'train': 0, 'val': 0, 'test': 0}

    # Placeholder for loading data
    def load_data(self):
        pass

    # Placeholder for setting training size
    def set_training_size(self, training_size=cfg.training_size):
        pass

    # Placeholder for getting model
    def get_model(self, model_name, split_layer=0):
        pass

    # Get the size of the training dataset
    def get_train_data_size(self):
        return self._training_sizes['train']

    # Get the size of the validation dataset
    def get_val_data_size(self):
        return self._training_sizes['val']

    # Get the size of the test dataset
    def get_test_data_size(self):
        return self._training_sizes['test']

    # Get the training dataset
    def get_train_data(self):
        return self.datasets['train']

    # Get the validation dataset
    def get_val_data(self):
        return self.datasets['val']

    # Get the test dataset
    def get_test_data(self):
        return self.datasets['test']

    # Get the training dataloader
    def get_train_dataloader(self):
        return self.dataloaders['train']

    # Get the validation dataloader
    def get_val_dataloader(self):
        return self.dataloaders['val']

    # Get the test dataloader
    def get_test_dataloader(self):
        return self.dataloaders['test']


class NMParty(Party):

    def __init__(self, name):
        super(NMParty, self).__init__(name)
        # Initialize configuration parameters
        self.random_seed = cfg.random_seed
        self.num_devices = cfg.num_devices
        self.num_group = cfg.num_group
        self.num_classes = cfg.num_classes
        self.num_layers = cfg.num_layers
        self.num_frames = cfg.num_frames
        self.num_rounds = cfg.num_rounds
        self.num_epochs = cfg.num_epochs
        self.num_head = cfg.num_head
        self.batch_size = cfg.batch_size
        self.image_size = cfg.image_size
        self.model_size = cfg.model_size
        self.cluster = cfg.cluster
        self.learning_rate = cfg.learning_rate
        self.iters = cfg.iters
        self.offload = cfg.offload
        self.model_name = cfg.model_name
        self.temperature = cfg.temperature
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.embed_dim = cfg.embed_dim
        self.threshold = cfg.threshold
        self.slide = cfg.slide
        self.root_dada = cfg.root_dada
        self.root_dada_json = cfg.root_dada_json
        self.TRAINED_FOLDER_STORAGE = cfg.TRAINED_FOLDER_STORAGE

        # Initialize models
        self.V_uninet = self.get_model(model_name=self.model_name[0])
        self.A_uninet = self.get_model(model_name=self.model_name[1])
        self.M_uninet = self.get_model(model_name=self.model_name[2])

    # Load data and apply transformations
    def load_data(self):
        train_transforms = T.create_video_transform(input_size=self.image_size, is_training=True, distortion=0.5,
                                                    hflip=0.5, color_jitter=None)
        val_transforms = T.create_video_transform(input_size=self.image_size, is_training=False)

        train_transforms_compress = [train_transforms, Tr.Compose([
            Tr.RandomHorizontalFlip(1),
            Tr.RandomErasing(1),
            Tr.RandomVerticalFlip(1),
            Tr.RandomRotation(30),
            Tr.GaussianBlur(1)
        ])]

        val_transforms_compress = [val_transforms, None]

        train_dataset = NEARMISS(
            annotate_file=[self.root_dada_json],
            root=[self.root_dada],
            data_type='training',
            num_frames=self.num_frames,
            transforms=train_transforms_compress,
            size=self.image_size,
            num_classes=self.num_classes,
            device=self.device
        )

        val_dataset = NEARMISS(
            annotate_file=[self.root_dada_json],
            root=[self.root_dada],
            data_type='validating',
            num_frames=self.num_frames,
            transforms=val_transforms_compress,
            size=self.image_size,
            num_classes=self.num_classes,
            device=self.device
        )

        test_dataset = NEARMISS(
            annotate_file=[self.root_dada_json],
            root=[self.root_dada],
            data_type='testing',
            num_frames=self.num_frames,
            transforms=val_transforms_compress,
            size=self.image_size,
            num_classes=self.num_classes,
            device=self.device
        )

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
        self.datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        self._training_sizes = {'train': len(self.datasets['train']), 'val': len(self.datasets['val']),
                                'test': len(self.datasets['test'])}

    # Set the size of training, validation, and test datasets
    def set_training_size(self, training_size=cfg.training_size):
        self._training_sizes = {
            'train': int(float(min(training_size['train'], len(self.datasets['train']))) / self.batch_size),
            'val': int(float(min(training_size['val'], len(self.datasets['val']))) / self.batch_size),
            'test': int(float(min(training_size['test'], len(self.datasets['test']))) / self.batch_size)
        }
        logger.info(self._training_sizes)

    # Get the model based on the location and model name
    def get_model(self, location='unit', model_name='', split_layer=0):
        if model_name == 'tfe':
            if location == 'unit':
                model = TFE_U(
                    split_layer=split_layer,
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_frames,
                    num_classes=self.num_classes
                )
            if location == 'client':
                model = TFE_C(
                    split_layer=split_layer,
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_frames,
                    num_classes=self.num_classes
                )
            if location == 'server':
                model = TFE_S(
                    split_layer=split_layer,
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_frames,
                    num_classes=self.num_classes
                )

        elif model_name == 'hbco':
            model = HBCO(in_channels=self.num_frames, out_channels=self.embed_dim)

        elif model_name == 's3d':
            model = S3D(num_classes=self.embed_dim)

        logger.debug(str(model))
        return model.to(self.device)


# Create NMParty object, load data, and set training size
def create_parties_NMParty_experiment(name='server'):
    Party = NMParty(name)
    Party.load_data()
    Party.set_training_size()
    return Party


# Federated averaging function to average weights
def fed_avg(ws):
    """
    Returns the average of the weights.
    """
    w = list(ws.values())
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# Split weights for the client
def split_weights_client(weights, cweights):
    keys = list(weights.keys())
    ckeys = list(cweights.keys())
    for i in range(len(ckeys)):
        assert cweights[ckeys[i]].size() == weights[keys[i]].size()
        cweights[ckeys[i]] = weights[keys[i]]
    return cweights


# Split weights for the server
def split_weights_server(weights, cweights, sweights):
    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)
    for i in range(len(skeys)):
        assert sweights[skeys[i]].size() == weights[keys[i + len(ckeys)]].size()
        sweights[skeys[i]] = weights[keys[i + len(ckeys)]]
    return sweights


# Concatenate client and server weights
def concat_weights(weights, cweights, sweights):
    concat_dict = collections.OrderedDict()
    ckeys = list(cweights)
    skeys = list(sweights)
    keys = list(weights)
    for i in range(len(ckeys)):
        concat_dict[keys[i]] = cweights[ckeys[i]]
    for i in range(len(skeys)):
        concat_dict[keys[i + len(ckeys)]] = sweights[skeys[i]]
    return concat_dict


# Calculate FLOPs (Floating Point Operations) for a model
def calculate_flops(gen, size=np.prod(cfg.image_size)):
    import re
    def flops_layer(layer, size):
        """
        Calculate the number of flops for given a string information of layer.
        We extract only reasonable numbers and use them.

        Args:
            layer (str): example
                Linear (512 -> 1000)
                Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        """
        idx_type_end = layer.find('(')
        type_name = layer[:idx_type_end]
        params = re.findall('[^a-z](\d+)', layer)
        flops = 1
        if layer.find('Linear') >= 0:
            C1 = int(params[0])
            C2 = int(params[1])
            flops = np.prod([C1, C2], dtype=np.int64)
        elif layer.find('Conv2d') >= 0:
            C1 = int(params[0])
            C2 = int(params[1])
            K1 = int(params[2])
            K2 = int(params[3])
            flops = np.prod([C1, C2, K1, K2, size], dtype=np.int64)
        elif layer.find('Conv3d') >= 0:
            C1 = int(params[0])
            C2 = int(params[1])
            K1 = int(params[2])
            K2 = int(params[3])
            K3 = int(params[4])
            flops = np.prod([C1, C2, K1, K2, K3, size], dtype=np.int64)
        return flops

    def calculate_flops_sum(gen, size):
        def get_num_gen(gen):
            return sum(1 for x in gen)

        def flops_layer(layer, size):
            """
            Calculate the number of flops for given a string information of layer.
            We extract only reasonable numbers and use them.

            Args:
                layer (str): example
                    Linear (512 -> 1000)
                    Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
            """
            idx_type_end = layer.find('(')
            type_name = layer[:idx_type_end]
            params = re.findall('[^a-z](\d+)', layer)
            flops = 1
            if layer.find('Linear') >= 0:
                C1 = int(params[0])
                C2 = int(params[1])
                flops = np.prod([C1, C2], dtype=np.int64)
            elif layer.find('Conv2d') >= 0:
                C1 = int(params[0])
                C2 = int(params[1])
                K1 = int(params[2])
                K2 = int(params[3])
                flops = np.prod([C1, C2, K1, K2, size], dtype=np.int64)
            elif layer.find('Conv3d') >= 0:
                C1 = int(params[0])
                C2 = int(params[1])
                K1 = int(params[2])
                K2 = int(params[3])
                K3 = int(params[4])
                flops = np.prod([C1, C2, K1, K2, K3, size], dtype=np.int64)
            return flops

        f = []
        for child in gen:
            num_children = get_num_gen(child.children())
            if num_children == 0:
                f.append(flops_layer(str(child), size))
            else:
                f.append(calculate_flops_sum(child.children(), size))
        return np.sum(f, dtype=np.int64)

    def get_num_gen(gen):
        return sum(1 for x in gen)

    f = []
    for child in gen:
        num_children = get_num_gen(child.children())
        if num_children == 0:
            f.append(flops_layer(str(child), size))
        else:
            f.append(calculate_flops_sum(child.children(), size))
    return f


# Calculate moving average of a list
def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


# Flatten a nested list
def flatten(x):
    flat_list = []
    for sublist in x:
        if isinstance(sublist, list):
            flat_list.extend(flatten(sublist))
        else:
            flat_list.extend([sublist])
    return flat_list


# Report memory usage and CPU temperature
def mem_report(overhead={'temp': 0, 'packet': 0, 'new_bytes': 0, 'total': 0, 'used': 0, 'free': 0}):
    G = 1000000000
    cpu = 0
    if cfg.SERVER_ADDR == '127.0.0.1':
        pass
    else:
        try:
            cpu = CPUTemperature().temperature
        except:
            pass
    total = round(psutil.virtual_memory().total / G, 3)
    used = round(psutil.virtual_memory().used / G, 3)
    free = round(psutil.virtual_memory().free / G, 3)
    new = (psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv) / G
    packet = round(new - overhead['new_bytes'], 5)
    return {'temp': cpu, 'packet': packet, 'new_bytes': new, 'total': total, 'used': used, 'free': free}


if __name__ == '__main__':
    Party = create_parties_NMParty_experiment(name='server')
