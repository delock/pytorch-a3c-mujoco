# pytorch-a3c-mujoco

<p><img src="asset/logo.png" align="middle"></p>

This repo is a fork of a3c for mujoco from https://github.com/andrewliao11/pytorch-a3c-mujoco , enabled for latest mujoco free version (210) and fixed some compatibility bugs.

This code aims to solve some control problems, espicially in Mujoco, and is highly based on [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c). What's difference between this repo and pytorch-a3c:

- compatible to Mujoco envionments
- the policy network output the mu, and sigma 
- construct a gaussian distribution from mu and sigma
- sample the data from the gaussian distribution
- modify entropy

Note that this repo is only compatible with Mujoco in OpenAI gym. If you want to train agent in Atari domain, please refer to [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c).

## Usage

A bash script is used to start workers for training, optimizer, and test(display)

```
./run_1_node.sh InvertedPendulum-v2
```

This project also defines it's own environements under env/ directory.   Train a custom inverted pendulum environment with
```
./run_1_node.sh InvertedPendulum-down
```

This is a much harder task because the pole is initially facing down, so the agent needs to learn to 'swing' the pole up then keep it there.  You can tweek this task by set a proper reward function.

You can also train a half cheetah
```
./run_1_node.sh HalfCheetah-v2
```

Or a humanoid
```
./run_1_node.sh Humanoid-v2
```

## Experiment results

Experiment result of this fork to be added later

### learning curve

The plot of total reward/episode length in 1000 steps:

To be added later

### video

To be added later

## Requirements

See INSTALL.md

## Reference
- [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
