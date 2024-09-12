import torch
import torch.nn as nn
from torch.autograd import Variable
from torchmetrics import Accuracy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
import numpy as np
import sys, os, copy, logging, warnings
import time
import socket
warnings.simplefilter(action="ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('../')

import config  as cfg
from events import Utils
from events.Comm import Communicator
from function.meter import AverageValueMeter
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
np.random.seed(cfg.random_seed)
torch.manual_seed(cfg.random_seed)

class Server(Communicator):
    def __init__(self, index, server_addr, server_port, limit_bandwidth, name='server'):
        super(Server, self).__init__(index, bandwidth=limit_bandwidth)
        self.server_addr, self.server_port = server_addr, server_port
        logger.info('Preparing Data.')
        self.Party = Utils.create_parties_NMParty_experiment(name=name)
        
        # %%%%%%%%%%%%%%%%%%%%%%%
        self.group_labels = {}
        self.weights = {c: None for c in self.Party.model_name}
        # %%%%%%%%%%%%%%%%%%%%%%%
        while True:
            self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            try:
                self.sock.bind((self.server_addr, self.server_port))
                self.sock.listen(self.Party.num_devices)
                break
            except BaseException:
                time.sleep(1)
                self.sock.close()
                continue
        # %%%%%%%%%%%%%%%%%%%%%%%
        
    def initialize(self, first, r=0):
        self.r = r
        self.first = first
        self.clear_memory()
        # %%%%%%%%%%%%%%%%%%%%%%%
        logger.info("Waiting Incoming Connections.")
        self.sock.settimeout(30)
        self.client_socks = {}
        while True:
            try:
                (client_sock, (ip, port)) = self.sock.accept()
                self.client_socks[str(ip)] = client_sock
                if len(self.client_socks) == self.Party.num_devices: break
            except socket.timeout:
                logger.info("No receive any new connection for {} seconds.".format(60))
                if len(self.client_socks) < self.Party.num_group:
                    continue
                else:
                    break

        self.client_socks = dict(sorted(self.client_socks.items(), key=lambda item: item[0]))
        logger.info(list(self.client_socks.keys()))

        # %%%%%%%%%%%%%%%%%%%%%%%
        logger.info("Initialize.")
        if self.first:
            ### FULL LAYERS
            self.split_layers = {c:self.Party.num_layers-1 for c in self.client_socks}  # Reset with full offloading
            ### BUILD MODEL
            self.V_uninet = self.Party.get_model(model_name=self.Party.model_name[0])
            self.A_uninet = self.Party.get_model(model_name=self.Party.model_name[1])
            self.M_uninet = self.Party.get_model(model_name=self.Party.model_name[2])
            self.nets = {c: self.Party.get_model('server',model_name=self.Party.model_name[2],split_layer=self.split_layers[c]) for c in self.client_socks}

            #### RELOAD
            self.V_path = os.path.join(self.Party.TRAINED_FOLDER_STORAGE,self.Party.model_name[0] + '.pth')
            self.A_path = os.path.join(self.Party.TRAINED_FOLDER_STORAGE,self.Party.model_name[1] + '.pth')
            self.M_path = os.path.join(self.Party.TRAINED_FOLDER_STORAGE,self.Party.model_name[2] + '.pth')
            
            if os.path.exists(self.V_path):
                weight = torch.load(self.V_path)
                if list(weight.keys()) == list(self.V_uninet.state_dict().keys()):
                    self.V_uninet.cpu().load_state_dict(weight)
            elif self.weights[self.Party.model_name[0]]:
                self.V_uninet.cpu().load_state_dict(self.weights[self.Party.model_name[0]])
            #
            if os.path.exists(self.A_path):
                weight = torch.load(self.A_path)
                if list(weight.keys()) == list(self.A_uninet.state_dict().keys()):
                    self.A_uninet.cpu().load_state_dict(weight)
            elif self.weights[self.Party.model_name[1]]:
                self.A_uninet.cpu().load_state_dict(self.weights[self.Party.model_name[1]])
            #
            if os.path.exists(self.M_path):
                weight = torch.load(self.M_path)
                if list(weight.keys()) == list(self.M_uninet.state_dict().keys()):
                    self.M_uninet.cpu().load_state_dict(weight)
            elif self.weights[self.Party.model_name[2]]:
                self.M_uninet.cpu().load_state_dict(self.weights[self.Party.model_name[2]])

            #### Backbone FLOPS

            self.list_flops_backbone_0 = np.sum(Utils.calculate_flops(list(self.V_uninet.children())),dtype=np.int64)
            self.list_flops_backbone_1 = np.sum(Utils.calculate_flops(list(self.A_uninet.children())),dtype=np.int64)
            self.list_flops_backbone = Utils.calculate_flops(list(self.M_uninet.feed_forward))
            self.list_flops_backbone_2 = np.sum(Utils.calculate_flops(list(self.M_uninet.children()))) - np.sum(self.list_flops_backbone)
            self.list_flops_backbone = list(map(lambda x:np.sum([x,self.list_flops_backbone_0,
                                                                 self.list_flops_backbone_1,
                                                                 self.list_flops_backbone_2]),self.list_flops_backbone))


        if self.Party.offload:
            ### SPLIT LAYERS
            if len(self.split_layers) != len(self.client_socks):
                self.split_layers = {c: self.Party.num_layers-1 for c in self.client_socks}
            self.scatter({c: ['SPLIT_LAYERS', self.split_layers[c]] for c in self.client_socks}, type=0)
        else:
            self.split_layers = {c: self.Party.num_layers - 1 for c in self.client_socks}
            self.scatter({c: ['SPLIT_LAYERS', self.split_layers[c]] for c in self.client_socks}, type=0)

        ### SPLIT MODEL
        for i, c in enumerate(self.client_socks):
            self.nets[c] = self.Party.get_model('server',model_name=self.Party.model_name[2],split_layer=self.split_layers[c])
            cweights = self.Party.get_model('client',model_name=self.Party.model_name[2],split_layer=self.split_layers[c]).cpu().state_dict()
            # offloading weight in server also need to be initialized from the same global weight
            sweights = Utils.split_weights_server(self.M_uninet.cpu().state_dict(), cweights, self.nets[c].cpu().state_dict())
            self.nets[c].load_state_dict(sweights)

        self.criterion_1 = nn.L1Loss().to(self.Party.device)
        self.criterion_2 = nn.BCELoss().to(self.Party.device)
        self.metric = Accuracy(task="binary").to(self.Party.device)
        self.optimizers = {c: torch.optim.Adam(list(self.A_uninet.parameters()) + list(self.nets[c].parameters()),
                                               lr=self.Party.learning_rate) for c in self.client_socks}
        self.lr_scheds = {c: torch.optim.lr_scheduler.MultiStepLR(self.optimizers[c], [50, 100]) for c in
                          self.client_socks}

        ### SEND GLOBAL WEIGHTS

        self.scatter(['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT',
                      [[self.V_uninet.cpu().state_dict(),self.A_uninet.cpu().state_dict(),self.M_uninet.cpu().state_dict()],
                       self.r, self.first, self.Party.iters]])

    def _thread_network_testing(self, client_ip):
        msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
        if msg[1] is not None:
            self.send_msg(self.client_socks[client_ip], ['MSG_TEST_NETWORK', [self.V_uninet.cpu().state_dict(),
                                                                              self.A_uninet.cpu().state_dict(),
                                                                              self.M_uninet.cpu().state_dict()]])

        msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TEST_NETWORK')
        self.bandwidths[client_ip], \
            self.overheads[client_ip], \
            self.Party.iters[client_ip] = [round(msg[1][0], 3), msg[1][1], int(msg[1][2])] if msg[1] is not None else (0.0, {}, 0)

    def _thread_training_time_per_iteration(self, client_ip):
        msg = self.recv_msg(self.client_socks[client_ip], 'MSG_TIME_PER_ITERATION')

        self.client_training_time[client_ip] = round(msg[1], 3) if msg[1] is not None else 0.0
        self.ttpis[client_ip] = round(msg[2], 3) if msg[1] is not None else 0.0

        self.client_testing_time[client_ip] = round(msg[3], 3) if msg[1] is not None else 0.0
        self.ptpis[client_ip] = round(msg[4], 3) if msg[1] is not None else 0.0

        self.train_activate_time[client_ip] = round(msg[5], 3) if msg[1] is not None else 0.0
        self.test_activate_time[client_ip] = round(msg[6], 3) if msg[1] is not None else 0.0

        self.client_infer_time[client_ip] = msg[7] if msg[1] is not None else 0.0

    def _thread_training_offloading(self, client_ip):
        self.nets[client_ip].train()
        self.nets[client_ip].to(self.Party.device)
        s_time_total = time.time()
        for epoch in range(self.Party.num_epochs):
            # logger.info("Step {} / {}".format(epoch+1, self.Party.num_epochs))
            # logger.info("-" * 10)
            loss_meters = {'loss': AverageValueMeter(), 'mae_loss': AverageValueMeter(),'bce_loss': AverageValueMeter()}
            metric_meters = {'accuracy': AverageValueMeter(),'fps':AverageValueMeter()}
            with torch.set_grad_enabled(True):
                idx = 0
                logs = {}
                while int(idx) < min(self.Party.get_train_data_size(), self.Party.iters[client_ip])-1:
                    
                    self.optimizers[client_ip].zero_grad()
                    msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                    idx = msg[1]
                    if idx is None: break
                    smashed_output, feat_vids, feat_hbco, targets, fps = msg[2]
                    feed_fwd_out, norm1_out, feat_vids, feat_hbco, targets, fps = \
                       Variable(smashed_output[0].to(self.Party.device), requires_grad=True), \
                        Variable(smashed_output[1].to(self.Party.device), requires_grad=True), \
                        Variable(feat_vids.to(self.Party.device), requires_grad=True), \
                        Variable(feat_hbco.to(self.Party.device), requires_grad=True), \
                        Variable(targets.to(self.Party.device)), \
                        Variable(fps.to(self.Party.device)), \

                    feat_output = self.nets[client_ip]([feed_fwd_out,norm1_out])

                    logits = torch.sigmoid(feat_output)
                    targets = targets.clamp(self.Party.temperature, 1 - self.Party.temperature)

                    # mean absolute error
                    mae_loss = self.criterion_1(feat_vids, feat_hbco)

                    # binary cross entropy
                    bce_loss = self.criterion_2(logits, targets)

                    loss = mae_loss + bce_loss
                    acc = self.metric(logits, (targets > cfg.threshold).long())

                    loss.backward()
                    self.optimizers[client_ip].step()

                    # Send gradients to client

                    self.send_msg(self.client_socks[client_ip], ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT', feed_fwd_out.grad.cpu()])

                    loss_meters['loss'].add(loss.cpu().detach().numpy())
                    loss_meters['mae_loss'].add(mae_loss.cpu().detach().numpy())
                    loss_meters['bce_loss'].add(bce_loss.cpu().detach().numpy())
                    metric_meters['accuracy'].add(acc.cpu().detach().numpy())
                    metric_meters['fps'].add(int(np.average(fps.cpu().detach().numpy())))
                    logs.update({k: np.round(v.mean, 4) for k, v in loss_meters.items()})
                    logs.update({k: np.round(v.mean, 4) if k!='fps' else int(v.mean) for k, v in metric_meters.items()})

                    self.client_train_metric[client_ip] = logs

                logger.info(f'[train][{client_ip}]:{logs}')
                self.lr_scheds[client_ip].step()
            # logger.info(str(client_ip) + ' offloading training end')

        e_time_total = time.time()
        self.server_training_time[client_ip] = round(e_time_total - s_time_total, 3)

    def _thread_testing(self, client_ip):

        s_time_total = time.time()
        for epoch in range(self.Party.num_epochs):
            # logger.info("Step {} / {}".format(epoch + 1, self.Party.num_epochs))
            # logger.info("-" * 10)
            loss_meters = {'loss': AverageValueMeter(), 'mae_loss': AverageValueMeter(),
                           'bce_loss': AverageValueMeter()}
            metric_meters = {'accuracy': AverageValueMeter(), 'fps': AverageValueMeter()}
            idx = 0
            logs = {}
            while int(idx) < min(self.Party.get_test_data_size(), self.Party.iters[client_ip]) - 1:

                msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                if msg[1] is None:
                    break
                else:
                    idx, (loss, mae_loss, bce_loss, acc, fps) = msg[1:]

                    loss_meters['loss'].add(loss.cpu().detach().numpy())
                    loss_meters['mae_loss'].add(mae_loss.cpu().detach().numpy())
                    loss_meters['bce_loss'].add(bce_loss.cpu().detach().numpy())
                    metric_meters['accuracy'].add(acc.cpu().detach().numpy())
                    metric_meters['fps'].add(int(np.average(fps.cpu().detach().numpy())))
                    logs.update({k: np.round(v.mean, 4) for k, v in loss_meters.items()})
                    logs.update(
                        {k: np.round(v.mean, 4) if k != 'fps' else int(v.mean) for k, v in metric_meters.items()})

                    self.client_test_metric[client_ip] = logs
            logger.info(f'[test][{client_ip}]:{logs}')

        e_time_total = time.time()
        self.server_testing_time[client_ip] = round(e_time_total - s_time_total, 3)

    def _thread_training_average(self, client_ip):
        msg = self.recv_msg(self.client_socks[client_ip], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER')
        if msg[1] is not None:
            self.w_local_list_V[client_ip] = msg[1][0]
            self.w_local_list_A[client_ip] = msg[1][1]
            self.w_local_list_M[client_ip] = Utils.concat_weights(self.M_uninet.cpu().state_dict(), msg[1][2],
                                                                self.nets[client_ip].cpu().state_dict())


    def clustering_kmean(self):
        try:
            X = np.array([[self.ttpis[c], self.bandwidths[c], self.offloading[c]] for c in self.client_socks]).astype(
                'float16')
            self.input_group = X
            # Clustering without network limitation
            component = self.Party.num_group if len(self.client_socks) >= self.Party.num_group else len(self.client_socks)
            self.kmeans = KMeans(n_clusters=component,
                                 random_state=self.Party.random_seed)

            pred = self.kmeans.fit_predict(X)

            self.group_labels = {c: pred[i] for i, c in enumerate(self.client_socks)}
        except ValueError as e:
            logger.info(e)

    def clustering_gmm(self):
        try:
            X = np.array(
                [[self.ttpis[c], self.bandwidths[c], self.offloading[c]] for i, c in enumerate(self.client_socks)]).astype(
                'float16')
            self.input_group =X
            # Clustering without network limitation
            component = self.Party.num_group if len(self.client_socks) >= self.Party.num_group else len(self.client_socks)
            self.gmm = GaussianMixture(n_components=component,
                                       covariance_type='full',
                                       random_state=self.Party.random_seed)

            pred = self.gmm.fit_predict(X)
            self.group_labels = {c: pred[i] for i, c in enumerate(self.client_socks)}
            self.state = self.gmm.means_.T.flatten()
            self.scale = (1 - MinMaxScaler().fit_transform(self.gmm.means_[:, 0].reshape(-1, 1))).flatten()
        except ValueError as e:
            logger.info(e)

    def aggregate(self):
        self.w_local_list_V = {}
        self.w_local_list_A = {}
        self.w_local_list_M = {}
        self.net_threads = {c: threading.Thread(target=self._thread_training_average, args=(c,)) for c in
                            self.client_socks}
        [self.net_threads[c].start() for c in self.client_socks]
        [self.net_threads[c].join() for c in self.client_socks]

        self.V_uninet.cpu().load_state_dict(Utils.fed_avg(self.w_local_list_V))
        self.A_uninet.cpu().load_state_dict(Utils.fed_avg(self.w_local_list_A))
        self.M_uninet.cpu().load_state_dict(Utils.fed_avg(self.w_local_list_M))

    def test(self):
        self.server_infer_time= {self.Party.model_name[0]:[],        #predict time per iterations
                    self.Party.model_name[1]:[],
                    self.Party.model_name[2]:[],
                    }

        for phase in ['test']:
            loss_meters = {'loss': AverageValueMeter(), 'mae_loss': AverageValueMeter(),'bce_loss': AverageValueMeter()}
            metric_meters = {'accuracy': AverageValueMeter(),'fps':AverageValueMeter()}
            if phase == 'test':
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
                    desc=f'[{phase}][{self.Party.name}]',
                    disable=False,
            ) as iterator:
                for idx, (vids, hbco, targets,fps) in enumerate(iterator):
                    if idx >= self.Party._training_sizes[phase]: break
                    logs = {}

                    with torch.set_grad_enabled(False):
                        vids, hbco, targets, fps = \
                            Variable(vids.to(self.Party.device), requires_grad=True), \
                            Variable(hbco.to(self.Party.device), requires_grad=True), \
                            Variable(targets.to(self.Party.device)), \
                            Variable(fps.to(self.Party.device)), \

                        t0 = time.time()
                        feat_vids = self.V_uninet(vids)
                        t1=time.time()
                        feat_hbco = self.A_uninet(hbco)
                        t2 = time.time()
                        feat_output = self.M_uninet([feat_hbco, feat_hbco, feat_vids])  # q, k, v
                        t3 = time.time()
                        logits = torch.sigmoid(feat_output)
                        targets = targets.clamp(self.Party.temperature, 1 - self.Party.temperature)

                        # mean absolute error
                        mae_loss = self.criterion_1(feat_vids, feat_hbco)

                        # binary cross entropy
                        bce_loss = self.criterion_2(logits, targets)

                        loss = mae_loss + bce_loss
                        acc = self.metric(logits, (targets > cfg.threshold).long())

                        self.server_infer_time[self.Party.model_name[0]].append(t1 - t0)
                        self.server_infer_time[self.Party.model_name[1]].append(t2 - t1)
                        self.server_infer_time[self.Party.model_name[2]].append(t3 - t2)
                        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                        loss_meters['loss'].add(loss.cpu().detach().numpy())
                        loss_meters['mae_loss'].add(mae_loss.cpu().detach().numpy())
                        loss_meters['bce_loss'].add(bce_loss.cpu().detach().numpy())
                        metric_meters['accuracy'].add(acc.cpu().detach().numpy())
                        metric_meters['fps'].add(int(fps.cpu().detach().numpy()))
                        logs.update({k: np.round(v.mean, 4) for k, v in loss_meters.items()})
                        logs.update({k: np.round(v.mean, 4) if k!='fps' else int(v.mean) for k, v in metric_meters.items()})
                        iterator.set_postfix_str(", ".join([f"{k}={v}" for k, v in logs.items()]))

            it =self.Party.batch_size*self.Party._training_sizes[phase]
            self.server_infer_time[self.Party.model_name[0]]= round(sum(self.server_infer_time[self.Party.model_name[0]])/it,3)
            self.server_infer_time[self.Party.model_name[1]]= round(sum(self.server_infer_time[self.Party.model_name[1]])/it,3)
            self.server_infer_time[self.Party.model_name[2]]= round(sum(self.server_infer_time[self.Party.model_name[2]])/it,3)

            ### sum(test time)/(image in batch <- "btach size" * total batch)

            self.server_test_metric = (logs)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.weights = {self.Party.model_name[0]:self.V_uninet.cpu().state_dict(),
                       self.Party.model_name[1]:self.A_uninet.cpu().state_dict(),
                       self.Party.model_name[2]:self.M_uninet.cpu().state_dict()}

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if cfg.tracker:
            torch.save(self.V_uninet.cpu().state_dict(), self.V_path)
            torch.save(self.A_uninet.cpu().state_dict(), self.A_path)
            torch.save(self.M_uninet.cpu().state_dict(), self.M_path)

    def reinitialize(self, first=True, r=0):
        self.initialize(first, r=r)

    def concat_norm(self):

        group_max_index = [0] * self.Party.num_group
        group_max_value = [0] * self.Party.num_group

        for i, (c, label) in enumerate(self.group_labels.items()):
            if self.ttpis[c] >= group_max_value[label]:
                group_max_value[label] = self.ttpis[c]
                group_max_index[label] = i

        ttpi_order = np.array([self.ttpis[i] for i in self.client_socks])[np.array(group_max_index)]
        bw_order = np.array([self.bandwidths[i] for i in self.client_socks])[np.array(group_max_index)]
        offloading_order = np.array([self.offloading[i] for i in self.client_socks])[np.array(group_max_index)]

        self.state = np.concatenate([ttpi_order,bw_order, offloading_order])

    def get_offloading(self):
        self.offloading = {}
        workload = 0
        assert len(self.split_layers) == len(self.client_socks)
        for c in self.client_socks:
            for l in range(self.Party.num_layers):
                if l <= self.split_layers[c]:
                    workload += self.list_flops_backbone[l]
            self.offloading[c] = round(workload / sum(self.list_flops_backbone), 3)
            workload = 0

    def expand_actions(self, actions):  # Expanding group actions to each device
        full_actions = []
        for i, c in enumerate(self.client_socks):
            full_actions.append(actions[self.group_labels[c]])
        return full_actions

    def action_to_layer(self, action):  # Expanding group actions to each device
        # first caculate cumulated flops
        model_state_flops = []
        cumulated_flops = 0

        for l in self.list_flops_backbone:
            cumulated_flops += l
            model_state_flops.append(cumulated_flops)

        model_flops_list = np.array(model_state_flops)
        model_flops_list = model_flops_list / cumulated_flops

        split_layer = []
        for v in action:
            idx = np.where(np.abs(model_flops_list - v) == np.abs(model_flops_list - v).min())
            idx = idx[0][-1]-1 if idx[0][-1]>=self.Party.num_layers-1 else idx[0][-1]
            split_layer.append(idx)

        return {c: split_layer[i] for i, c in enumerate(self.client_socks)}

    def action_offload(self, actions):
        if self.Party.cluster == 'gmm':
            self.action = self.expand_actions((actions[0] + self.scale) / 2)
            self.action_mean = self.expand_actions((actions[1] + self.scale) / 2)
        else:
            self.action = self.expand_actions(actions[0])
            self.action_mean = self.expand_actions(actions[1])

        self.std = self.expand_actions(actions[2])
        self.split_layers = self.action_to_layer(self.action)

        if self.Party.offload:
            logger.info(f'State: ' + str(self.state))
            logger.info('Current action mean: ' + str(self.action_mean))
            logger.info('Current action sample: ' + str(self.action))
            logger.info('Current standard: ' + str(self.std))
            logger.info(f'Group index - {self.Party.cluster}: ' + str(self.group_labels))
            logger.info('Current OPs index: ' + str(self.split_layers))
        else:
            self.action = [1.0 for i in self.client_socks]
            self.action_mean = [1.0 for i in self.client_socks]
            self.std = [1.0 for i in self.client_socks]
    def scatter(self, msg, type=1):
        if type:
            [self.send_msg(self.client_socks[c], msg) for c in self.client_socks]
        else:
            [self.send_msg(self.client_socks[c], msg[c]) for c in self.client_socks]

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            pass

    def calculate_reward(self):
        import operator
        reward = 0
        max_ttpi_full_offload = max(self.ttpi_full_offload.items(), key=operator.itemgetter(1))[1]
        max_ttpi_offload = max(self.ttpi_offload.items(), key=operator.itemgetter(1))[1]
        max_ttpi_offload_index = max(self.ttpi_offload.items(), key=operator.itemgetter(1))[0]
        if list(self.ttpi_full_offload.keys()) == list(self.ttpi_offload.keys()):
            if self.Party.cluster == 'kmean':
                ## offload < baseline
                for k in self.ttpi_offload:
                    if self.ttpi_offload[k] < self.ttpi_full_offload[k]:
                        r = (self.ttpi_full_offload[k] - self.ttpi_offload[k]) / (
                            self.ttpi_full_offload[k] if self.ttpi_full_offload[k] else 1)
                        reward += r
                    else:
                        r = (self.ttpi_offload[k] - self.ttpi_full_offload[k]) / (
                            self.ttpi_offload[k] if self.ttpi_offload[k] else 1)
                        reward -= r
            else:
                reward = np.average(
                    np.array(list(self.ttpi_full_offload.values())) - np.array(list(self.ttpi_offload.values())))

            done = 1 if max_ttpi_offload > max_ttpi_full_offload else 0
        else:
            done = 1

        if self.Party.offload:
            logger.info('Current reward: ' + str(reward))
            logger.info('Current maxtime: ' + str(max_ttpi_offload))
            logger.info('Terminal: ' + str(done))
        else:
            self.ttpi_offload = self.ttpi_full_offload

        return reward, max_ttpi_offload, max_ttpi_offload_index, done

    def run(self):
        s_time = time.time()

        self.server_test_metric = []
        self.server_train_metric = []
        self.client_test_metric = {c: {} for c in self.client_socks}
        self.client_train_metric = {c: {} for c in self.client_socks}

        self.Party.iters = {c: 0 for c in self.client_socks}
        self.bandwidths = {c: 0 for c in self.client_socks}
        self.client_training_time = {c: 0 for c in self.client_socks}
        self.client_testing_time = {c: 0 for c in self.client_socks}
        self.server_training_time = {c: 0 for c in self.client_socks}
        self.server_testing_time = {c: 0 for c in self.client_socks}
        self.train_activate_time = {c: 0 for c in self.client_socks}
        self.test_activate_time = {c: 0 for c in self.client_socks}
        self.client_infer_time = {c: 0 for c in self.client_socks}
        self.ttpis = {c: 0 for c in self.client_socks}  # Training time per iteration
        self.ptpis = {c: 0 for c in self.client_socks}  # predicted time per iteration
        self.overheads = {c: {} for c in self.client_socks}

        ####################################
        self.net_threads = {c: threading.Thread(target=self._thread_network_testing, args=(c,)) for c in
                            self.client_socks}
        [self.net_threads[c].start() for c in self.client_socks]
        [self.net_threads[c].join() for c in self.client_socks]

        ####################################
        self.scatter(['Start'])
        ##

        # Training start

        self.net_threads = {c: threading.Thread(target=self._thread_training_offloading, args=(c,)) for c in
                            self.client_socks}
        [self.net_threads[c].start() for c in self.client_socks]
        [self.net_threads[c].join() for c in self.client_socks]

        self.net_threads = {c: threading.Thread(target=self._thread_testing, args=(c,)) for c in
                            self.client_socks}
        [self.net_threads[c].start() for c in self.client_socks]
        [self.net_threads[c].join() for c in self.client_socks]

        self.net_threads = {c: threading.Thread(target=self._thread_training_time_per_iteration, args=(c,)) for c in
                            self.client_socks}
        [self.net_threads[c].start() for c in self.client_socks]
        [self.net_threads[c].join() for c in self.client_socks]

        ####################################
        self.scatter(['Finish'])
        ##
        ### COMPUTE
        thread_time = time.time() - s_time
        self.get_offloading()
        if self.Party.cluster == 'kmean':
            self.clustering_kmean()
            self.concat_norm()
        else:
            self.clustering_gmm()
        ### AVERAGE MODEL
        self.aggregate()

        self.test()

        return self.state, self.overheads, self.bandwidths, self.ttpis, self.ptpis, \
            self.server_training_time, self.client_training_time, self.server_testing_time, self.client_testing_time, \
            self.server_infer_time, self.client_infer_time, \
            self.train_activate_time, self.test_activate_time,\
            self.input_group, self.group_labels, \
            self.client_train_metric, self.client_test_metric, \
            self.server_train_metric, self.server_test_metric, \
            thread_time

    def reset(self, r=0):
        self.reset_flag = True

        s_time_rebuild = time.time()
        self.reinitialize(self.reset_flag, r=r)
        e_time_rebuild = time.time()
        rebuild_time = round(e_time_rebuild - s_time_rebuild, 3)
        logger.info('ROUND: {} START'.format(r))
        state, overheads, bandwidths, ttpis, ptpis, \
        server_training_time, client_training_time, server_testing_time, client_testing_time, \
        server_infer_time, client_infer_time, \
        train_activate_time, test_activate_time, \
        input_group, group_labels, \
        client_train_metric, client_test_metric, \
        server_train_metric, server_test_metric, thread_time = self.run()

        self.ttpi_full_offload = ttpis

        total_time = rebuild_time + thread_time

        logger.info('<<<<<<<< ==== RESET ==== >>>>>>>> \n')
        logger.info('Bandwidth: ' + str(bandwidths))
        logger.info('Training time per iteration: ' + str(ttpis))
        logger.info('client infer time: ' + str(client_infer_time))
        logger.info('server infer time: ' + str(server_infer_time))
        logger.info('Client train time: ' + str(client_training_time))
        logger.info('Server train time: ' + str(server_training_time))
        logger.info('Activate train time: ' + str(train_activate_time))
        logger.info('Client test time: ' + str(client_training_time))
        logger.info('Server test time: ' + str(server_training_time))
        logger.info('Activate test time: ' + str(train_activate_time))

        logger.info('Rebuild time: ' + str(rebuild_time))
        logger.info('Thread time: ' + str(thread_time))
        logger.info('Total time: ' + str(total_time))
        logger.info('==> Reinitialization for Episodes : {:}'.format(r))
        logger.info('ROUND: {} END'.format(r))
        return state

    def set(self, actions, r=0):

        self.reset_flag = False
        self.action_offload(actions)
        s_time_rebuild = time.time()
        self.reinitialize(self.reset_flag, r=r)
        e_time_rebuild = time.time()
        rebuild_time = round(e_time_rebuild - s_time_rebuild, 3)
        logger.info('ROUND: {} START'.format(r))
        state, overheads, bandwidths, ttpis, ptpis, \
        server_training_time, client_training_time, server_testing_time, client_testing_time, \
        server_infer_time, client_infer_time, \
        train_activate_time, test_activate_time, \
        input_group, group_labels, \
        client_train_metric, client_test_metric, \
        server_train_metric, server_test_metric, thread_time = self.run()

        self.ttpi_offload = ttpis

        reward, maxtime, maxtime_index, done = self.calculate_reward()
        total_time = rebuild_time + thread_time

        logger.info('<<<<<<<< ==== SET ==== >>>>>>>> \n')
        logger.info('Bandwidth: ' + str(bandwidths))
        logger.info('Training time per iteration: ' + str(ttpis))
        logger.info('client infer time: ' + str(client_infer_time))
        logger.info('server infer time: ' + str(server_infer_time))
        logger.info('Client train time: ' + str(client_training_time))
        logger.info('Server train time: ' + str(server_training_time))
        logger.info('Activate train time: ' + str(train_activate_time))
        logger.info('Client test time: ' + str(client_training_time))
        logger.info('Server test time: ' + str(server_training_time))
        logger.info('Activate test time: ' + str(train_activate_time))

        logger.info('Rebuild time: ' + str(rebuild_time))
        logger.info('Thread time: ' + str(thread_time))
        logger.info('Total time: ' + str(total_time))
        logger.info('==> Reinitialization for Episodes : {:}'.format(r+1))
        logger.info('ROUND: {} END'.format(r))
        return reward, maxtime, maxtime_index, done, state, overheads, bandwidths, ttpis, ptpis, \
        server_training_time, client_training_time, server_testing_time, client_testing_time, \
        server_infer_time, client_infer_time, \
        train_activate_time, test_activate_time, input_group, group_labels, \
        client_train_metric, client_test_metric, server_train_metric, server_test_metric, \
        thread_time, rebuild_time, total_time,\
        self.split_layers, self.action, self.action_mean, self.std

