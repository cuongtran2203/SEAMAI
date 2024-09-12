##


Here's the rewritten README with improved clarity and structure:


## Setup
rm -rf ${HOME}/.cache/pip
pip3 uninstall torch torchinfo torchmetrics torchview torchvision pytorchvideo -y
pip3 uninstall nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 -y

pip3 install -r requirements.txt
pip3 install torch==2.1 torchvision==0.16 torchview==0.2.6 --index-url https://download.pytorch.org/whl/cu118
pip3 install torchmetrics==0.11.4 pytorchvideo==0.1.5 torchinfo==1.8.0


## Run

To start the client and server for real training outside the global folder, follow these steps:

1. Run `client.py` and wait until you see the debug message: "Waiting Incoming Connections."
2. Run `server.py`. If `listen` equals 30, the client will break and run; otherwise, it will wait for the next client for 60 seconds.

## Config

1. Note `SERVER_ADDR`, `SERVER_PORT`, and `SOURCE_ADDR`.
2. Save results in the `results` folder.
3. Adjust any hyper-parameters as needed:
    - For training parameters, see the `Party` class in `Utils`.
    - For global configurations, refer to `cfg`.
4. Update the paths as required.
5. Set `offload = True` for Reinforcement Learning, or `offload = False` for basic Federated Learning (FL).
6. `num_devices` should match the number of `SOURCE_ADDR` (waiting for more clients can take a long time).
7. Ensure `num_group` is less than or equal to `num_devices`.
8. Use `cluster = 'gmm'` (considering three items: ttpi, bw, flop) (AOP algorithm) or `cluster = 'kmean'` (FedAdapt algorithm).
9. Framework operations may take longer if `limit_bw` is set.
10. To maximize any value for data or bandwidth, set it to `float('inf')`.

## Dataset

1. Note the paths for the dataset.
2. Use the `GB_nearmiss` folder for full data.
3. Use `GB_nearmiss.json` for labeling each video in the `nearmiss` folder and for Heartbeat, CO sensing.
4. The dataloader will handle item retrieval with preprocessing.
5. The sample dataset processed.
6. Participants can adjust the data to achieve the best results, for example using data balancing methods, data augmentation, etc.

## Events <AOP algorithm at here>

1. `Client` pool manages client operations.
2. `Server` pool manages server operations.
3. `Comm`   pool handles data transformation (send, recv).
4. `Agent`  pool manages the Agent for Reinforcement Learning.
5. `Utils`  pool handles dataloader and utility functions, focusing on the `Party` class.

## Models

Models are created for:
- `hbco`: sensor data
- `s3d`: video data
- `tfe`: transformer encoder

## Function

The `meter` function calculates the mean metric after each training batch.

## Training

Save the training weight in results/train (tfe.pth, hbco.pth, tfe.pth)
The results of training process saved it on results/train/AOP_res.npy
If you want to extract some thing what you want, please used it (or write the function to plot it, for example: time, accuracy, loss, etc)
Remember that if you want to run the other traing phase, firstly please remove AOP_res.npy then, run the other simulations.
The total training time of AOP should be faster than the two cases, FedAvg and FL, and as fast as can be achieved. How to enable/disenable FedAvg and FL are clearly indicated in the source code.

## Testing

- Use the inference.py function for testing phase
- Participants can adjust parameter settings for the highest possible accuracy (from about 70-80%).
