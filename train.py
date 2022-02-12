import math
import os
import sys
import time
import os.path

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from test import test
import pdb


def setup(args):
    # logging
    log_dir = os.path.join('logs', args.model_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_filename = args.env_name+"."+args.model_name+".log"
    f = open(os.path.join(log_dir, log_filename), "w")
    # model saver
    ckpt_dir = os.path.join('ckpt', args.model_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    ckpt_filename = args.env_name+"."+args.model_name+".pkl"
    return (f, os.path.join(ckpt_dir, ckpt_filename)), (log_dir, ckpt_dir)

def get_checkpoint_name(args, i):
    ckpt_dir = os.path.join('ckpt', args.model_name)
    ckpt_filename = args.env_name+"."+args.model_name+"."+str(i)+".pkl"
    return os.path.join(ckpt_dir, ckpt_filename)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def save_grads(model, rank):
    filename = 'grads-{}.pt'.format(rank)
    while os.path.exists(filename):
        pass
    grads = []
    for param in model.parameters():
        grads.append(param.grad)
    torch.save(grads, 'sync/_{}'.format(filename))
    os.rename('sync/_{}'.format(filename), 'sync/{}'.format(filename))

def load_grads(model, rank):
    filename = 'sync/grads-{}.pt'.format(rank)
    if not os.path.exists(filename):
        return False
    grads = torch.load(filename)
    i = 0
    for param in model.parameters():
        #if param.grad is not None:
        #    print ('Model grad is None')
        #    return
        param._grad = grads[i]
        i += 1
    os.remove(filename)
    return True

def optimize_loop(world_size, model, optimizer):
    t0 = time.time()
    optimize_iter = 0

    # sync the initial model
    torch.save(model.state_dict(), 'sync/_model.pt')
    os.rename('sync/_model.pt', 'sync/model.pt')

    while True:
        has_update = False
        for i in range(world_size):
            optimizer.zero_grad()
            if load_grads(model, i):
                has_update = True
                optimizer.step()
                optimize_iter += 1
                if optimize_iter % 100 == 0:
                    t1 = time.time()
                    print ("optimized {} iterations, throughput = {} iter/s                    ".format(optimize_iter, int(optimize_iter/(t1-t0)*100)/100), end="\r")
        if has_update:
            torch.save(model.state_dict(), 'sync/_model.pt')
            os.rename('sync/_model.pt', 'sync/model.pt')

def load_model(model):
    while not os.path.exists('sync/model.pt'):
        pass
    model.load_state_dict(torch.load('sync/model.pt'))

# global variable pi
pi = Variable(torch.FloatTensor([math.pi]))
def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

def train_loop(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)
    test_env = create_atari_env(args.env_name)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()
    (f, ckpt_path), (log_dir, ckpt_dir) = setup(args)

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    iteration = 0
    t_load = 0
    t_save = 0
    t_learn = 0
    t0 = time.time()
    while True:
        tb0 = time.time()
        episode_length += 1
        # Sync with the shared model every iteration
        load_model(model)
        tb1 = time.time()
        if done:
            # initialization
            cx = Variable(torch.zeros(1, 128))
            hx = Variable(torch.zeros(1, 128))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            # for mujoco, env returns DoubleTensor
            value, mu, sigma_sq, (hx, cx) = model(
                (Variable(state.float().unsqueeze(0).float()), (hx, cx)))
            sigma_sq = F.softplus(sigma_sq)
            eps = torch.randn(mu.size())
            # calculate the probability
            action = (mu + sigma_sq.sqrt()*Variable(eps)).data
            prob = normal(action, mu, sigma_sq)
            entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

            entropies.append(entropy)
            log_prob = prob.log()

            state, reward, done, _ = env.step(action[0].numpy())
            # prevent stuck agents
            done = done or episode_length >= args.max_episode_length
            # reward shaping
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _, _ = model((Variable(state.float().unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        # calculate the rewards from the terminal state
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            # convert the data into xxx.data will stop the gradient
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            # for Mujoco, entropy loss lower to 0.0001
            policy_loss = policy_loss - (log_probs[i]*Variable(gae).expand_as(log_probs[i])).sum() \
                                        - (0.0001*entropies[i]).sum()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        tb2 = time.time()
        save_grads(model, rank)
        tb3 = time.time()
        t_load += tb1-tb0
        t_save += tb3-tb2
        t_learn += tb2-tb1
        t_total = t_load+t_save+t_learn
        if rank == 0 and iteration % args.save_freq == 0:
            t1 = time.time()
            duration = t1 - t0
            torch.save(model.state_dict(), get_checkpoint_name(args, iteration))
            if iteration == 0:
                print ("Saving checkpoint of iteration {}, value_loss = {}, policy_loss = {}.".format(iteration, value_loss[0][0], policy_loss))
            else:
                print ("Saving checkpoint of iteration {}, value_loss = {}, policy_loss = {}, throughput = {} load:save:learn={}%:{}%:{}%.".format(iteration, value_loss[0][0], policy_loss, int(args.save_freq/duration*10)/10, int(t_load/t_total*100), int(t_save/t_total*100), int(t_learn/t_total*100)))
            if args.display:
                test(rank, args, model, test_env)
            t0 = t1
        iteration += 1
