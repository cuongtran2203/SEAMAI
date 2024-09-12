import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchmetrics import Accuracy
from tqdm import tqdm
import time
import socket
import numpy as np
import sys, os, logging, warnings,random
warnings.simplefilter(action="ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('../')

import config as cfg
from events import Utils
from events.Comm import Communicator
from function.meter import AverageValueMeter

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
np.random.seed(cfg.random_seed)
torch.manual_seed(cfg.random_seed)

class Client(Communicator):
    def __init__(self, index, server_addr, server_port, limit_bandwidth, name='client',
                 source_addr = socket.gethostbyname(socket.gethostname()), source_port=0):
        super(Client, self).__init__(index, bandwidth = limit_bandwidth[source_addr])
        self.server_addr, self.server_port = server_addr, server_port
        self.source_addr,self.source_port=  source_addr,source_port
        logger.info('Preparing Data.')
        self.Party = Utils.create_parties_NMParty_experiment(name=name)
        self.Party.device = cfg.set_devices[source_addr]
        # %%%%%%%%%%%%%%%%%%%%%%%
        self.overhead = Utils.mem_report()

    def initialize(self, first, r=0):
        self.r = r
        self.first = first
        self.clear_memory()
        # %%%%%%%%%%%%%%%%%%%%%%%
        logger.info("Waiting Incoming Connections.")
        self.sock.close()
        while True:
            self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.sock.bind((self.source_addr,self.source_port))
            try:
                self.sock.connect((self.server_addr, self.server_port))
                logger.info('Connecting to Server.')
                break
            except BaseException:
                time.sleep(1)
                self.sock.close()
                continue
        # %%%%%%%%%%%%%%%%%%%%%%%
        logger.info("Initialize.")
        if self.first:
            ### FULL LAYERS
            self.split_layer = self.Party.num_layers-1
            ### BUILD MODEL
            self.V_uninet = self.Party.get_model(model_name=self.Party.model_name[0])
            self.A_uninet = self.Party.get_model(model_name=self.Party.model_name[1])
            self.M_uninet = self.Party.get_model(model_name=self.Party.model_name[2])
            self.net = self.Party.get_model('client',model_name=self.Party.model_name[2])

        ### SPLIT LAYERS
        msg = self.recv_msg(self.sock, 'SPLIT_LAYERS')
        self.split_layer = int(msg[1] if msg[1] is not None else self.Party.num_layers-1)

        ### SPLIT MODEL
        self.net = self.Party.get_model('client',model_name=self.Party.model_name[2],split_layer=self.split_layer)

        self.criterion_1 = nn.L1Loss().to(self.Party.device)
        self.criterion_2 = nn.BCELoss().to(self.Party.device)
        self.metric = Accuracy(task="binary").to(self.Party.device)
        self.optimizer = torch.optim.SGD(list(self.A_uninet.parameters())+list(self.net.parameters()), lr=self.Party.learning_rate)
        self.lr_sched = optim.lr_scheduler.MultiStepLR(self.optimizer, [50, 100])

        ### RECV GLOBAL WEIGHTS
        msg = self.recv_msg(self.sock, 'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT')
        if msg[1] is not None:
            weights, self.r, self.first, self.Party.iters = msg[1]
            self.V_uninet.cpu().load_state_dict(weights[0])
            self.A_uninet.cpu().load_state_dict(weights[1])
            self.M_uninet.cpu().load_state_dict(weights[2])
            self.net.cpu().load_state_dict(Utils.split_weights_client(weights[2], self.net.cpu().state_dict()))



    def train(self):
        self.overhead = Utils.mem_report(self.overhead)

        logger.info('overhead: ' + str(self.overhead))
        # %%%%%%%%%%%%%%%%%%%%%%%
        # Network speed test
        network_time_start = time.time()
        self.send_msg(self.sock, ['MSG_TEST_NETWORK', [self.V_uninet.cpu().state_dict(),
                                                       self.A_uninet.cpu().state_dict(),
                                                       self.M_uninet.cpu().state_dict()]])
        self.recv_msg(self.sock, 'MSG_TEST_NETWORK')
        network_time_end = time.time()

        if cfg.set_bw[self.source_addr] < float('inf'):
            network_speed = round(random.uniform(cfg.set_bw[self.source_addr] - 10, cfg.set_bw[self.source_addr]), 3)
        else:
            network_speed = (2 * self.Party.model_size * 8) / (network_time_end - network_time_start)  # Mbps
        logger.info('Network speed is {:}'.format(network_speed))

        self.send_msg(self.sock, ['MSG_TEST_NETWORK', [network_speed, self.overhead, self.Party.iters[self.source_addr]]])
        # %%%%%%%%%%%%%%%%%%%%%%%
        # Training start
        inference_time = {self.Party.model_name[0]:0,        #predict time per iterations
                    self.Party.model_name[1]:0,
                    self.Party.model_name[2]:0,
                    }
        if self.recv_msg(self.sock, 'Start'):
            s_time_total = time.time()
            train_activate_time = 0
            test_activate_time = 0
            for epoch in range(self.Party.num_epochs):
                for phase in ['train','test']:
                    if phase == 'train':
                        self.V_uninet.eval()
                        self.A_uninet.train()

                        self.net.train()
                        self.V_uninet.to(self.Party.device)
                        self.A_uninet.to(self.Party.device)
                        self.net.to(self.Party.device)

                    elif phase == 'test':
                        self.V_uninet.eval()
                        self.A_uninet.eval()
                        self.M_uninet.eval()
                        self.V_uninet.to(self.Party.device)
                        self.A_uninet.to(self.Party.device)
                        self.M_uninet.to(self.Party.device)
                        # %%%%%%%%%%%%

                    with tqdm(
                            self.Party.dataloaders[phase],
                            total=self.Party._training_sizes[phase],
                            desc=f'[{phase}][{self.Party.name}][epoch={epoch + 1}]',
                            disable=False,
                    ) as iterator:

                        for idx, (vids, hbco, targets, fps) in enumerate(iterator):
                            if idx > min(self.Party._training_sizes[phase],self.Party.iters[self.source_addr])-1: break

                            with torch.set_grad_enabled(phase == 'train'):

                                vids, hbco, targets, fps = \
                                    Variable(vids.to(self.Party.device), requires_grad=True), \
                                    Variable(hbco.to(self.Party.device), requires_grad=True), \
                                    Variable(targets.to(self.Party.device)), \
                                    Variable(fps.to(self.Party.device)), \

                                if phase =='train':
                                    self.optimizer.zero_grad()


                                    feat_vids = self.V_uninet(vids)
                                    feat_hbco = self.A_uninet(hbco)
                                    smash_output = self.net([feat_hbco, feat_hbco, feat_vids])  # q, k, v

                                    t0=time.time()
                                    self.send_msg(self.sock, ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', idx,
                                                              ([smash_output[0].cpu(),smash_output[1].cpu()],
                                                               feat_vids.cpu(),feat_hbco.cpu(),targets.cpu(),fps.cpu())])
                                    # Wait receiving server gradients
                                    msg = self.recv_msg(self.sock, 'MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT')
                                    gradients = msg[1].to(self.Party.device) if msg[1] is not None else msg[1]
                                    smash_output[0].backward(gradients)
                                    train_activate_time += time.time() - t0


                                    self.optimizer.step()
                                    self.lr_sched.step()

                                elif phase == 'test':

                                    t0a = time.time()
                                    feat_vids = self.V_uninet(vids)
                                    t1a = time.time()
                                    feat_hbco = self.A_uninet(hbco)
                                    t2a = time.time()
                                    feat_output = self.M_uninet([feat_hbco, feat_hbco, feat_vids])  # q, k, v
                                    t3a = time.time()
                                    logits = torch.sigmoid(feat_output)
                                    targets = targets.clamp(self.Party.temperature, 1 - self.Party.temperature)
                                    # mean absolute error
                                    mae_loss = self.criterion_1(feat_vids, feat_hbco)
                                    # binary cross entropy
                                    bce_loss = self.criterion_2(logits, targets)
                                    loss = mae_loss + bce_loss
                                    acc = self.metric(logits, (targets > cfg.threshold).long())

                                    t0 = time.time()
                                    self.send_msg(self.sock, ['MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER', idx, (loss.cpu(),
                                                                                                              mae_loss.cpu(),
                                                                                                              bce_loss.cpu(),
                                                                                                              acc.cpu(),
                                                                                                              fps.cpu())])
                                    test_activate_time += time.time() - t0
                    e_time_total = time.time()
                    if phase == 'train':
                        training_time = e_time_total - s_time_total
                        logger.info('Total time: ' + str(e_time_total - s_time_total))
                        training_time_pr = (e_time_total - s_time_total) / np.ceil(min(self.Party.get_train_data_size(), self.Party.iters[self.source_addr]))
                        logger.info('training_time_per_iteration: ' + str(training_time_pr))
                    elif phase == 'test':
                        testing_time = e_time_total - s_time_total
                        logger.info('Total time: ' + str(e_time_total - s_time_total))
                        testing_time_pr = (e_time_total - s_time_total) / np.ceil(min(self.Party.get_test_data_size(), self.Party.iters[self.source_addr]))
                        logger.info('testing_time_per_iteration: ' + str(testing_time_pr))

                        inference_time[self.Party.model_name[0]]=round(t1a - t0a, 3)
                        inference_time[self.Party.model_name[1]]=round(t2a - t1a, 3)
                        inference_time[self.Party.model_name[2]]=round(t3a - t2a, 3)


            ##############
            ##############

            self.send_msg(self.sock, ['MSG_TIME_PER_ITERATION', training_time, training_time_pr,
                                      testing_time, testing_time_pr,
                                      train_activate_time, test_activate_time,
                                      inference_time])

            ### new weight
            self.upload()

            return training_time+testing_time

    def upload(self):
        msg = self.recv_msg(self.sock, 'Finish')
        if msg[0] is not None:
            self.send_msg(self.sock, ['MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER', [self.V_uninet.cpu().state_dict(),
                                                                             self.A_uninet.cpu().state_dict(),
                                                                             self.net.cpu().state_dict()]])

    def reinitialize(self, first, r=0):
        self.initialize(first,r=r)

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            pass

    def set(self):
        s_time_rebuild = time.time()
        self.reinitialize(self.first)
        e_time_rebuild = time.time()
        if self.r + 1:
            logger.info('ROUND: {} START'.format(self.r))
            training_time = self.train()
            if self.first:
                logger.info('<<<<<<<< ==== RESET ==== >>>>>>>> \n')
                logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
                logger.info('Total Time: {:}'.format(training_time))
                logger.info('==> Reinitialization for ROUND : {:}'.format(self.r))

            else:
                logger.info('<<<<<<<< ==== SET ==== >>>>>>>> \n')
                logger.info('Rebuild time: ' + str(e_time_rebuild - s_time_rebuild))
                logger.info('Total Time: {:}'.format(training_time))
                logger.info('==> Reinitialization for ROUND : {:}'.format(self.r+1))
        logger.info('ROUND: {} END'.format(self.r))


