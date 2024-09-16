import torch
import cv2
import os
import numpy as np
from models.tfe import *
import config as cfg
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
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
from torch.autograd import Variable
from torchmetrics import Accuracy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScale

'''
Define dataloader
'''


 # Load data and apply transformations
def load_data():
    train_transforms = T.create_video_transform(input_size=cfg.image_size, is_training=True, distortion=0.5,
                                                hflip=0.5, color_jitter=None)
    val_transforms = T.create_video_transform(input_size=cfg.image_size, is_training=False)

    train_transforms_compress = [train_transforms, Tr.Compose([
        Tr.RandomHorizontalFlip(1),
        Tr.RandomErasing(1),
        Tr.RandomVerticalFlip(1),
        Tr.RandomRotation(30),
        Tr.GaussianBlur(1)
    ])]

    val_transforms_compress = [val_transforms, None]

    train_dataset = NEARMISS(
        annotate_file=[cfg.root_dada_json],
        root=[cfg.root_dada],
        data_type='training',
        num_frames=cfg.num_frames,
        transforms=train_transforms_compress,
        size=cfg.image_size,
        num_classes=cfg.num_classes,
        device= device
    )

    val_dataset = NEARMISS(
        annotate_file=[cfg.root_dada_json],
        root=[cfg.root_dada],
        data_type='validating',
        num_frames=cfg.num_frames,
        transforms=val_transforms_compress,
        size=cfg.image_size,
        num_classes=cfg.num_classes,
        device=device
    )

    test_dataset = NEARMISS(
        annotate_file=[cfg.root_dada_json],
        root=[cfg.root_dada],
        data_type='testing',
        num_frames=cfg.num_frames,
        transforms=val_transforms_compress,
        size=cfg.image_size,
        num_classes=cfg.num_classes,
        device=device
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_dataloader,val_dataloader,test_dataloader


# Define model and tool train


class MM_sensing():
    def __init__(self,device) -> None:
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
        self.temperature = cfg.temperature
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
        self.device = device

        # Initialize models
        self.V_uninet = self.get_model(model_name=self.model_name[0])
        self.A_uninet = self.get_model(model_name=self.model_name[1])
        # self.M_uninet = self.get_model(model_name=self.model_name[2])
        self.fea =TFE_C(
                    split_layer=split_layer,
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_frames,
                    num_classes=self.num_classes
                )
        self.combine =  model = TFE_S(
                    split_layer=split_layer,
                    num_layers=self.num_layers,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_frames,
                    num_classes=self.num_classes
                )
        
        self.criterion_1 = nn.L1Loss().to(self.device)
        self.criterion_2 = nn.BCELoss().to(self.device)
        self.metric = Accuracy(task="binary").to(self.device)
        self.optimizers = {c: torch.optim.Adam(list(self.A_uninet.parameters()) + list(self.nets[c].parameters()),
                                               lr=self.Party.learning_rate) for c in self.client_socks}
        self.lr_scheds = {c: torch.optim.lr_scheduler.MultiStepLR(self.optimizers[c], [50, 100]) for c in
                          self.client_socks}
    
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
    
    def train(self,trainloader):
        Total_loss = 0
        MAE_loss = 0
        bce_Loss = 0
        ACC_train = 0
        for batch in tqdm(trainloader):
            vid,hbco,label,fps = batch
            fead_vid = self.V_uninet(vid)
            fead_hbco = self.A_uninet(hbco)
            out = self.fea([fead_hbco,fead_hbco,fead_vid])
            prediction = self.combine(out)
            logits = torch.sigmoid(prediction)
            label = label.clamp(self.temperature, 1 - self.temperature)
                                # mean absolute error
            mae_loss = self.criterion_1(fead_vid, fead_hbco)

            # binary cross entropy
            bce_loss = self.criterion_2(logits, label)

            loss = mae_loss + bce_loss
            acc = self.metric(logits, (label > cfg.threshold).long())
            loss.backward()
            self.optimizers.step()
            Total_loss += loss.item()
            MAE_loss+= mae_loss.item()
            bce_Loss += bce_loss.item()
            ACC_train += acc.item()
        Total_loss = Total_loss/len(trainloader)
        MAE_loss = MAE_loss/len(trainloader)
        bce_Loss = bce_Loss/len(trainloader)
        ACC_train = ACC_train/len(trainloader)
        return Total_loss,MAE_loss,bce_Loss,ACC_train
    def validate(self,valoader):
        self.A_uninet.eval()
        self.V_uninet.eval()
        self.fea.eval()
        self.combine.eval()
        ACC_val = 0
        Total_loss = 0
        Mae_loss = 0
        bce_loss = 0
        for batch in tqdm(valoader):
            vid,hbco,label,fps = batch
            fead_vid = self.V_uninet(vid)
            fead_hbco = self.A_uninet(hbco)
            out = self.fea([fead_hbco,fead_hbco,fead_vid])
            prediction = self.combine(out)
            logits = torch.sigmoid(prediction)
            label = label.clamp(self.temperature, 1 - self.temperature)
            acc = self.metric(logits, (label > cfg.threshold).long())
            mae_loss = self.criterion_1(fead_vid, fead_hbco)
            # binary cross entropy
            bce_loss = self.criterion_2(logits, label)
            loss = mae_loss + bce_loss
            ACC_val += acc.item()
            Total_loss += loss.item()
            Mae_loss += mae_loss.item()
            
        Total_loss = Total_loss/len(valoader)
        Mae_loss = Mae_loss/len(valoader)
        bce_Loss = bce_Loss/len(valoader) 
        ACC_val = ACC_val/len(valoader)
        return Total_loss,Mae_loss,bce_Loss,ACC_val
    
    
if __name__ == "__main__":
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import logging

    # Create and configure logger
    logging.basicConfig(filename="newfile.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Creating an object
    logger = logging.getLogger()
    writer = SummaryWriter()
    train_dataloader,val_dataloader,test_dataloader = load_data()
    Epochs = 100
    model = MM_sensing(device)
    
    for e in range(Epochs):
        Total_loss,MAE_loss,bce_Loss,ACC_train = model.train(train_dataloader)
        logger.info(f"AT {e} epoch ==> Total loss train: {Total_loss} MAE loss train: {MAE_loss} BCE loss train: {bce_Loss} ACC train: {ACC_train}")
        writer.add_scalar("Total loss train",Total_loss,e)
        writer.add_scalar("MAE loss train",MAE_loss,e)
        writer.add_scalar("BCE loss train",bce_Loss,e)
        writer.add_scalar("Accuracy train",ACC_train,e)
        Total_loss,Mae_loss,bce_Loss,ACC_val = model.validate(val_dataloader)
        writer.add_scalar("Total loss validate",Total_loss,e)
        writer.add_scalar("MAE loss validate",Mae_loss,e)
        writer.add_scalar("BCE loss validate",bce_Loss,e)
        writer.add_scalar("Accuracy validate",ACC_val,e)
        logger.info(f"AT {e} epoch ==> Total loss val: {Total_loss} MAE loss val: {Mae_loss} BCE loss val: {bce_Loss} ACC val: {ACC_val}")
