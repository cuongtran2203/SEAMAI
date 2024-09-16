#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys,os
import torch
sys.path.append('../')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tracker = True # Record the results and model for each update epoch
# Network configuration (do not change any things for SERVER_ADDR and SERVER_PORT
SERVER_ADDR = '10.23.2.120'
SERVER_PORT = 9371
# Unique clients order
# Do not change the SOURCE_ADDR (we fixed)
SOURCE_ADDR = {
    0: '10.23.2.121',1: '10.23.2.122',2: '10.23.2.123', 3: '10.23.2.124', 4: '10.23.2.125',
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
limit_bw = {c:v for (c,v) in zip(list(SOURCE_ADDR.values()), [
    float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),
                                                              ])}
                                #Mbps "Megabytes per second"
# Bandwidth for simulation (participant can be changed these value if you want)
set_bw = {c:v for (c,v) in zip(list(SOURCE_ADDR.values()), [
    20, 50, 80, 100, 200,
                                                              ])}
# Do not change the the gpu/cpu (we fixed)
set_devices = {c:v for (c,v) in zip(list(SOURCE_ADDR.values()), [
    'cuda','cuda','cpu','cpu','cpu',
                                                        ])}
# Participant can be changed these value if you want)
iters = {c:v for (c,v) in zip(list(SOURCE_ADDR.values()), [
    5,5,10,15,15,

                                                              ])}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
infer = 'ppo'                             # type of algorithm to RL
cluster = 'gmm'                           # type of embedding to cluster GMM = Gaussian Mixture Models
num_group = 3                             # Number of groups
num_devices = len(SOURCE_ADDR)            # Number of devices
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Model configuration (participant can be changed the tfe structure model if you want)
model_name = ['s3d','hbco','tfe']         # three models (s3d for video, hbco for IoT data sensors, tfe = transformer encoder for RL)
training_size = {'train':float('inf'), 'val': float('inf'),'test':5}
model_size = 31.68 + 6.37 + 21.39    # MB
num_layers = 10                           # number of TransformerEncoder  (multihead-attention)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Hyper-parameter
num_rounds = 100
num_epochs = 1
batch_size = 1
learning_rate = 0.01

## V - video
image_size = (224,224)
num_frames = 16
slide = 100
num_classes = 2

## A - attention
in_channels = 16
out_channels= 768

## M - multihead-attention
num_head  = 16
embed_dim = 768

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RL training configuration
offload = True
max_episodes = 80          # max training episodes
exploration_times = 5	   # exploration times without std decay
action_std = 0.5           # constant std for action distribution (Multivariate Normal)
update_timestep = 10       # update policy every n timestep
K_epochs = 50              # update policy for K epochs
eps_clip = 0.2             # clip parameter for PPO
rl_gamma = 0.9             # discount factor
rl_lr = 0.001              # parameters for Adam optimizer
rl_betas = (0.9, 0.999)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# dataset configuration
project_name = 'bRenchmark' # Project name
home_path = os.path.join(sys.path[0])
#train
root_dada = os.path.join(home_path, "dataset",'GB_nearmiss') # Path dataset
root_dada_json = os.path.join(home_path, "dataset", "GB_nearmiss.json")
#inference
#root_dada = os.path.join(home_path, "dataset",'GB_nearmiss_test') # Path dataset
#root_dada_json = os.path.join(home_path, "dataset",'GB_nearmiss_test', "GB_nearmiss_test.json")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# result path
save_result = os.path.join(home_path,"results")
TRAINED_FOLDER_STORAGE = os.path.join(save_result,"train")
if not os.path.exists(save_result): os.mkdir(save_result)
if not os.path.exists(TRAINED_FOLDER_STORAGE): os.mkdir(TRAINED_FOLDER_STORAGE)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temperature = 0.2 # Used to soften the targets
threshold = 0.5
random_seed=0
random = True
