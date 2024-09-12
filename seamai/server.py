import datetime

import torch
import sys, logging, os, warnings

warnings.simplefilter(action="ignore", category=UserWarning)
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append('../')

from events.Server import Server
from events.Agent import PPO, Memory
from events.Utils import moving_avg
import config as cfg

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

def initialize(first=True):
    logger.info('Preparing Sever.')
    # Creating environment
    server = Server(0, cfg.SERVER_ADDR, cfg.SERVER_PORT, cfg.limit_bw)
    server.initialize(first=first)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    state_dim = 3 * cfg.num_group
    action_dim = cfg.num_group
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if cfg.infer == 'ppo':
        # Creating PPO agent
        agent = PPO(state_dim, action_dim,
                    cfg.action_std, cfg.rl_lr,
                    cfg.rl_betas, cfg.rl_gamma,
                    cfg.K_epochs, cfg.eps_clip)

    memory = Memory()
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if first:
        if os.path.exists(os.path.join(cfg.TRAINED_FOLDER_STORAGE, f'{cfg.infer}.pth')):
            agent.policy.load_state_dict(torch.load(os.path.join(cfg.TRAINED_FOLDER_STORAGE, f'{cfg.infer}.pth')))
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if not os.path.exists(os.path.join(cfg.save_result, 'AOP_res.npy')):

            res = { 'client_training_time': [], 'server_training_time': [],
                    'client_testing_time': [], 'server_testing_time': [],
                    'train_activate_time': [], 'test_activate_time': [],

                    'server_infer_time': [], 'client_infer_time': [],
                    'rebuild_time': [],'thread_time': [],'total_time': [],

                    'client_test_acc': [], 'client_test_loss': [],
                    'server_test_acc': [], 'server_test_loss': [],

                    'server_train_metric': [], 'client_train_metric': [],
                    'server_test_metric': [], 'client_test_metric': [],

                    'bandwidths': [], 'input_group': [], 'group_labels': [],

                    'ttpis': [],'ptpis': [], 'max_ttpi': [], 'max_ttpi_index': [],

                    'action': [], 'action_mean': [], 'action_mean_smooth': [], 'action_group': [],
                    'split_layers': [], 'overheads': [], 'rewards': [], 'std': [], 'terminals': [], 'state': []}

            last_R = 0
        else:
            res = np.load(os.path.join(cfg.save_result, 'AOP_res.npy'), allow_pickle=True).tolist()
            logger.info('RELOAD')
            last_R = len(res['total_time']) + 1

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    first = True
    terminal = 1

    return server, agent, state_dim, action_dim, memory, first, terminal, res, last_R


# RL training
def main():
    server, agent, state_dim, action_dim, memory, first, terminal, res, r = initialize(first=True)
    update_epoch = r
    state = np.array([0] * state_dim)
    action_mean_full_smooth = [0.5] * action_dim
    while r < cfg.num_rounds:

        logger.info('\n#####################################################################\n')

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if terminal:  state = server.reset(r=r)  # ttpi, bw, flops of group
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        actions = agent.select_action(state, memory) # -> action -> flops -> split index

        reward, maxtime, maxtime_index, terminal, state, overheads, bandwidths, ttpis, ptpis,\
        server_training_time, client_training_time, server_testing_time, client_testing_time, \
        server_infer_time, client_infer_time, \
        train_activate_time, test_activate_time, input_group, group_labels, \
        client_train_metric, client_test_metric, server_train_metric, server_test_metric, \
        thread_time, rebuild_time, total_time, split_layers, action_full, action_mean_full, std_full = server.set(actions, r=r)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(terminal)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Update
        if not ((r + 1) % cfg.update_timestep) and not ((actions[0] == actions[1]).all()):
            agent.update(memory, r=r)
            if update_epoch >= cfg.exploration_times: agent.explore_decay(r - cfg.exploration_times)
            update_epoch += 1

        else:
            if cfg.offload:
                logger.info('[Current action mean smoothing] ' + str(action_mean_full_smooth))
                if not ((actions[0] == actions[1]).all()):
                    logger.info('[STAGE 1 - RL training] ====================================>')
                else:
                    terminal = 1 if first else 0
                    logger.info('[STAGE 2 - OP optimal]  ====================================>')
                    first = False
            else:
                logger.info('[STAGE 0 - FL training] ====================================>')
                terminal = 0
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if not ((r + 1) % cfg.max_episodes):
            terminal = 1
            memory.clear_memory()

        res['state'].append(state)
        res['action_group'].append(actions)
        res['action'].append(action_full)
        res['action_mean'].append(action_mean_full)
        x = [np.array(res['action_mean'])[:, i] for i in range(len(action_mean_full))]
        action_mean_full_smooth = [moving_avg(x[i], n=len(x[i]))[-1] for i in range(len(action_mean_full))]
        res['action_mean_smooth'].append(action_mean_full_smooth)
        res['std'].append(std_full)

        res['rewards'].append(reward)
        res['max_ttpi'].append(maxtime)
        res['max_ttpi_index'].append(maxtime_index)
        res['terminals'].append(terminal)

        res['rebuild_time'].append(rebuild_time)
        res['thread_time'].append(thread_time)
        res['total_time'].append(total_time)

        res['client_training_time'].append(client_training_time)
        res['server_training_time'].append(server_training_time)

        res['client_testing_time'].append(client_testing_time)
        res['server_testing_time'].append(server_testing_time)

        res['train_activate_time'].append(train_activate_time)
        res['test_activate_time'].append(test_activate_time)

        res['client_infer_time'].append(client_infer_time)
        res['server_infer_time'].append(server_infer_time)

        res['bandwidths'].append(bandwidths)
        res['ttpis'].append(ttpis)
        res['ptpis'].append(ptpis)
        res['overheads'].append(overheads)

        res['server_train_metric'].append(server_train_metric)
        res['client_train_metric'].append(client_train_metric)
        res['server_test_metric'].append(server_test_metric)
        res['client_test_metric'].append(client_test_metric)

        res['server_test_loss'].append(server_test_metric['loss'])
        res['server_test_acc'].append(server_test_metric['accuracy'] * 100)
        res['client_test_loss'].append(round(np.average([v['loss'] for k, v in client_test_metric.items()]),3))
        res['client_test_acc'].append(round(np.average([v['accuracy'] for k, v in client_test_metric.items()]),3)*100)

        res['input_group'].append(input_group)
        res['group_labels'].append(group_labels)
        res['split_layers'].append(split_layers)

        r += 1

        if cfg.tracker:
            # Record the results for each update epoch
            np.save(os.path.join(cfg.save_result, 'AOP_res.npy'), np.array(res))
            # save the agent every updates
            torch.save(agent.policy.state_dict(), os.path.join(cfg.TRAINED_FOLDER_STORAGE, f'{cfg.infer}.pth'))
    else:
        server.reinitialize(r=-1)
        logger.info('ROUND: {} END'.format(-1))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
