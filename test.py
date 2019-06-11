import argparse
import random                # Handling random number generation
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
from visdom import Visdom
import RL.helper as helper
import itertools
from functools import reduce
import matplotlib.pyplot as plt
from collections import deque

import gym
from RL.wrappers import VideoMonitor, Reset
from os import path
from RL.PPO.model_ppo import PPO

EXP_NAME = "exp-ppo-{}".format(datetime.now().strftime("%H:%M:%S"))
GLOBAL_STEP = 0
P_LOSSES = []
RESET_FRAME = torch.zeros([1, 4, 34, 136])
HISTOGRAM = {0:0, 1:0, 2:0, 3:0}
VIS = Visdom()


def __pars_args__():
    parser = argparse.ArgumentParser(description='PPO')

    parser.add_argument('-gam', '--gamma', type=float, default=0.95, help='discounting factor')
    parser.add_argument('-lam', '--lam', type=float, default=0.99, help='discounting factor')
    parser.add_argument('-m_grad', '--max_grad_norm', type=float, default=0.5, help='discounting factor')
    parser.add_argument('-ent_coef', '--ent_coef', type=float, default=0.,
                        help='policy entropy coefficient in the optimization objective')
    parser.add_argument('-vf_coef', '--vf_coef', type=float, default=0.5,
                        help='value function loss coefficient in the optimization objective')
    parser.add_argument("-tot_t", "--total_timesteps", type=int, default=10000000,
                        help="number of timesteps (i.e. number of actions taken in the environment)")

    parser.add_argument('-n_train_ep', '--number_train_epoch', type=int, default=4,
                        help='number of training epochs per update')
    parser.add_argument('-clp_rng', '--clip_range', type=float, default=0.2,
                        help='clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the ' +
                             'training and 0 is the end of the training')


    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size used during learning')
    parser.add_argument('-mini_bs', '--mini_batchs', type=int, default=4, help='number of training minibatches per update. For recurrent policies, should be smaller or equal than number of environments run in parallel.')
    parser.add_argument('-n_step', '--n_step', type=int, default=1024, help='number of steps of the vectorized environment per update')

    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./monitor', help='Path to save monitor of agent')
    parser.add_argument('-v', '--version', default='0', help='Path to save monitor of agent')

    parser.add_argument('-save_every', '--save_every', type=int, default=10,
                        help='number of timesteps between saving events')

    parser.add_argument('-log_every', '--log_every', type=int, default=10,
                        help='number of timesteps between logs events')


    parser.add_argument("--max_steps", type=int, default=500, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[56, 84], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=True, help="Use cuda")

    return parser.parse_args()


def eval_fn(model, obs):
    model.eval()
    with torch.set_grad_enabled(False):
        obs = torch.tensor(obs).float()
        value_f, action, neg_log_prob, entropy =model(obs)
        return value_f, action, neg_log_prob



def build_env(args, env_name=EXP_NAME):
    env = gym.make("CartPole-v0")
    # env = ToTorchObs(env)
    env = Reset(env)
    env = VideoMonitor(env, "videos", mode="evaluation")
    return env


if __name__ == '__main__':
    args = __pars_args__()
    device = torch.device("cpu")

    # test_environment(args, img_trs)
    env = build_env(args)

    # model param
    obs_size = reduce((lambda x, y: x * y), env.observation_space.shape)
    action_space = env.action_space.n

    checkpoint = torch.load(path.join("model", "0", "train_net.cptk"), map_location="cpu")

    # used for sampling
    model = PPO(obs_size, args.hidden_dim, action_space, n_step=args.n_step)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    ep_info_buf = deque(maxlen=100)

    obs = np.expand_dims(env.reset(), 0)
    done = False
    # Here, we init the lists that will contain the mb of experiences
    mb_obs, mb_rewards, mb_actions, mb_values, mb_done, mb_neg_log_prob = [], [], [], [], [], []
    ep_infos = []

    count = 0

    # For n in range number of steps
    while not done and count < args.max_steps:
        # Given observations, get action value and neglopacs
        # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
        value_f, action, neg_log_prob = eval_fn(model, obs)
        action = action.cpu().item()
        neg_log_prob = neg_log_prob.cpu().item()
        value_f = value_f.cpu().item()

        # Take actions in env and look the results
        # Infos contains a ton of useful informations
        obs, rewards, done, info = env.step(action)
        obs = np.expand_dims(obs, 0)
        count += 1

    print(count)


    env.close()
