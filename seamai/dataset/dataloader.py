import copy, random, time, warnings, logging
import torch
import torch.utils.data as data_utl
from tqdm import tqdm
import numpy as np
import json
import os, sys
import cv2
warnings.simplefilter(action="ignore", category=Warning)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('../')

import config as cfg
from torch.utils.data.sampler import WeightedRandomSampler
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# 2 classes
# 0 : Nearmiss
# 1 : Normal

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def make_dataset(annotate_file, type, root, num_frames=16, slide=100):
    dataset = []

    for it, base_path in enumerate(root):
        for jt, anno_path in enumerate(annotate_file):
            with open(anno_path, 'r') as f:
                data = json.load(f)
                for vid in tqdm(data.keys()):

                    if data[vid]['type'] not in type or not os.path.exists(os.path.join(base_path, vid)): continue

                    # TODO: video
                    video_path = os.path.join(base_path, vid)

                    # TODO: sensor

                    hb = float(data[vid]['heartbeat'])
                    co2 = float(data[vid]['co2'])

                    # TODO: label
                    label= int(data[vid]['label'])


                    frame_count = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

                    for start in range(0, frame_count - (num_frames + 1), slide):
                        for i in range(start,start+num_frames):
                            dataset.append((video_path, start, start + num_frames, hb, co2, label))


    return dataset

class NEARMISS(data_utl.Dataset):

    def __init__(self, annotate_file=[''], root='', data_type='training',
                 num_frames=32,  transforms=None, slide = 100,
                 size=(224, 224), num_classes=1,  device='cpu'):

        self.data = make_dataset(annotate_file, data_type, root, num_frames, slide)
        self.annotate_file = annotate_file
        self.transforms_video,self.transforms_sensor = transforms
        self.root = root
        self.size = size
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.device= device
        self.i=0
    def calulate_frame_per_seccond(self):
        self.new_frame_time = time.time()
        # Processing time for this frame = Current time – time when previous frame processed
        # fps will be number of frame processed in given time frame
        # since their will be time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (self.new_frame_time - self.prev_frame_time + 1e-5)
        self.prev_frame_time = copy.copy(self.new_frame_time)

        # converting the fps into integer
        fps = int(fps)
        return fps

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        self.index =index
        video_path, start_frame, end_frame, hb, co2, label = self.data[index]
        cam = cv2.VideoCapture(video_path)
        # used to record the time when we processed last frame
        self.prev_frame_time = 0
        # used to record the time at which we processed current frame
        self.new_frame_time = 0
        frames, fpss, hbs, co2s = [], [], [], []

        count = 0
        while True:
            ret, frame = cam.read()
            if ret:
                if count >= start_frame and count < end_frame:
                    img = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                    img = (img / 255.) * 2 - 1
                    frames.append(img)
                    fpss.append(self.calulate_frame_per_seccond())
                    hbs.append(hb)
                    co2s.append(co2)
                count += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()

        # [_,T,C,H,W] - [B,16,3,224,224]
        vids = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        # [_,T,C,H,W] - [B,1,16,2]

        Xs = np.array(hbs).astype(np.float32)
        mn, mx = 40, 120
        hbs = (Xs - mn) / (mx - mn)

        Xs = np.array(co2s).astype(np.float32)
        mn, mx = 600, 1500
        co2s = (Xs - mn) / (mx - mn)

        hbco = np.stack([hbs, co2s], axis=1).astype(np.float32)
        hbco = torch.from_numpy(hbco).unsqueeze(0)

        if self.transforms_video is not None:
            vids = self.transforms_video(vids.permute(0, 3, 1, 2)).permute(1, 0, 2, 3)
        if self.transforms_sensor is not None:
            hbco = self.transforms_sensor(hbco)



        if label==0:
            label = torch.tensor([1, 0], dtype=torch.float)
        else:
            label = torch.tensor([0, 1], dtype=torch.float)

        fps = torch.tensor(int(np.average(fpss)), dtype=torch.long)

        return vids.to(self.device), hbco.to(self.device), label.to(self.device), fps.to(self.device)

    def __len__(self):
        return len(self.data)
