from __future__ import print_function

import argparse
import os
import sys
import pickle as pkl

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import env
from envs import create_atari_env
from model import ActorCritic
from train import train_loop, optimize_loop
from test import test_loop
import my_optim
import pdb

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='required for batch.a3c (default: 128)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--model_name', type=str, default='a3c', 
                    help='used to save log file and model (default: a3c)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--world_size', type=int, default=1,
                    help='number of workers')
parser.add_argument('--rank', type=int, default=0,
                    help='rank of this worker')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v3', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')
parser.add_argument('--no-shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')
parser.add_argument('--display', type=bool, default=False, 
                    help='whether to use monitor and render ot not (default:False)')
parser.add_argument('--save_freq', type=int, default=20, 
                    help='how many intervals to save teh model (default:20)')
parser.add_argument('--task', choices=['train', 'eval', 'develop', 'optimize'], default='train', 
                    help='if use multi thread to train (default:True)')
parser.add_argument('--load_ckpt', type=str, default='ckpt/a3c/InvertedPendulum-v1.a3c.0.pkl')
parser.add_argument('--name', type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.name == None:
        args.name = args.env_name
    env = create_atari_env(args.env_name)

    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)

    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    if args.task == 'train':
        train_loop(args.rank, args, shared_model, optimizer)
    elif args.task == 'eval':
        test_loop(args)
    elif args.task == 'develop':
        train_loop(-1, args, shared_model, optimizer)
    elif args.task == 'optimize':
        optimize_loop(args.world_size, shared_model, optimizer)
