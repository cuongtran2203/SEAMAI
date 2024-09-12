import os
import sys
import time
import warnings
import logging

#for testing phase (inference)

import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score,  Precision, Recall
import torch.optim as optim
from torch.autograd import Variable

import config as cfg

sys.path.append('../')

# Suppress specific warnings
warnings.simplefilter(action="ignore", category=Warning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project-specific modules
from events import Utils
from function.meter import AverageValueMeter
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize Party
Party = Utils.create_parties_NMParty_experiment()
V_path = os.path.join(Party.TRAINED_FOLDER_STORAGE, Party.model_name[0] + '.pth')
A_path = os.path.join(Party.TRAINED_FOLDER_STORAGE, Party.model_name[1] + '.pth')
M_path = os.path.join(Party.TRAINED_FOLDER_STORAGE, Party.model_name[2] + '.pth')
if os.path.exists(V_path):
    weight = torch.load(V_path)
    if list(weight.keys()) == list(Party.V_uninet.state_dict().keys()):
        Party.V_uninet.cpu().load_state_dict(weight)
        print('reload')

if os.path.exists(A_path):
    weight = torch.load(A_path)
    if list(weight.keys()) == list(Party.A_uninet.state_dict().keys()):
        Party.A_uninet.cpu().load_state_dict(weight)
        print('reload')

    #
if os.path.exists(M_path):
    weight = torch.load(M_path)
    if list(weight.keys()) == list(Party.M_uninet.state_dict().keys()):
        Party.M_uninet.cpu().load_state_dict(weight)
        print('reload')

def main():
    res = []

    # Define loss functions and metrics
    criterion_1 = nn.L1Loss().to(Party.device)
    criterion_2 = nn.BCEWithLogitsLoss().to(Party.device)
    metric = Accuracy(task="binary").to(Party.device)
    metric1 = F1Score(task="binary").to(Party.device)
    metric2 = Precision(task="binary").to(Party.device)
    metric3 = Recall(task="binary").to(Party.device)

    # Set models to evaluation mode
    Party.V_uninet.eval()
    Party.A_uninet.eval()
    Party.M_uninet.eval()
    Party.V_uninet.to(Party.device)
    Party.A_uninet.to(Party.device)
    Party.M_uninet.to(Party.device)

    for phase in ['test']:
        loss_meters = {'loss': AverageValueMeter(), 'mae_loss': AverageValueMeter(), 'bce_loss': AverageValueMeter()}
        metric_meters = {
            'accuracy': AverageValueMeter(), 'f1score': AverageValueMeter(),
            'precision': AverageValueMeter(), 'recall': AverageValueMeter(),
            'fps': AverageValueMeter()
        }
        inference_time = {
            'vid_time': AverageValueMeter(), 'hbco_time': AverageValueMeter(),
            'tfe_time': AverageValueMeter(), 'loss_time': AverageValueMeter(),
            'total_time': AverageValueMeter()
        }

        with tqdm(Party.dataloaders[phase], total=Party._training_sizes[phase], desc=f'[{phase}][{Party.name}]',
                  disable=False) as iterator:
            for idx, (vids, hbco, targets, fps) in enumerate(iterator):
                logs = {}
                with torch.set_grad_enabled(False):
                    vids = Variable(vids.to(Party.device), requires_grad=True)
                    hbco = Variable(hbco.to(Party.device), requires_grad=True)
                    targets = Variable(targets.to(Party.device))
                    fps = Variable(fps.to(Party.device))

                    # Measure inference times
                    t0 = time.time()
                    feat_vids = Party.V_uninet(vids)
                    t1 = time.time()
                    feat_hbco = Party.A_uninet(hbco)
                    t2 = time.time()
                    feat_output = Party.M_uninet([feat_hbco, feat_hbco, feat_vids])  # q, k, v
                    t3 = time.time()
                    logits = torch.sigmoid(feat_output)
                    targets = targets.clamp(Party.temperature, 1 - Party.temperature)

                    # Compute losses and metrics
                    mae_loss = criterion_1(feat_vids, feat_hbco)
                    bce_loss = criterion_2(logits, targets)
                    loss = mae_loss + bce_loss
                    acc = metric(logits, (targets > 0.5).long())
                    f1 = metric1(logits, (targets > 0.5).long())
                    precision = metric2(logits, (targets > 0.5).long())
                    recall = metric3(logits, (targets > 0.5).long())
                    t4 = time.time()

                # Update loss meters
                loss_meters['loss'].add(loss.cpu().detach().numpy())
                loss_meters['mae_loss'].add(mae_loss.cpu().detach().numpy())
                loss_meters['bce_loss'].add(bce_loss.cpu().detach().numpy())
                metric_meters['accuracy'].add(acc.cpu().detach().numpy())
                metric_meters['f1score'].add(f1.cpu().detach().numpy())
                metric_meters['precision'].add(precision.cpu().detach().numpy())
                metric_meters['recall'].add(recall.cpu().detach().numpy())
                metric_meters['fps'].add(np.average(fps.cpu().detach().numpy()))

                # Update inference time meters
                inference_time['vid_time'].add(round(t1 - t0, 3))
                inference_time['hbco_time'].add(round(t2 - t1, 3))
                inference_time['tfe_time'].add(round(t3 - t2, 3))
                inference_time['loss_time'].add(round(t4 - t3, 3))
                inference_time['total_time'].add(round(t4 - t0, 3))

                logs.update({k: np.round(v.mean, 4) for k, v in loss_meters.items()})
                logs.update({k: np.round(v.mean, 4) if k != 'fps' else int(v.mean) for k, v in metric_meters.items()})
                logs.update({k: np.round(v.mean, 4) for k, v in inference_time.items()})
                iterator.set_postfix_str(", ".join([f"{k}={v}" for k, v in logs.items()]))

        res.append(logs)
        np.save(os.path.join(cfg.save_result, 'inference_results.npy'), np.array(res))


if __name__ == '__main__':
    # 1 sample = [16 frames, each frame (224,224,3) video] + [1 frame (16,2) sensor (Heartbeat & CO2)]
    main()
