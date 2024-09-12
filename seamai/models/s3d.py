# Total params: 7,920,298
# Trainable params: 7,920,298
# Non-trainable params: 0


import sys

sys.path.append('../')
import torch
from torchvision import models
import torch.nn as nn
from torchinfo import summary as model_summary

def S3D(fine_tune=True, num_classes=10):
    model = models.video.s3d()

    if fine_tune:
        # print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    if not fine_tune:
        # print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    model.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    return model

if __name__ == "__main__":
    vid = torch.ones([ 1,3,16, 224, 224])
    V = S3D()
    print(V(vid).shape)
    model_summary(V)

    size_model = 0
    for param in V.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")




