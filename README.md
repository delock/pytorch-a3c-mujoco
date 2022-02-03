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

There're three tasks/modes for you: train, eval, develop.

At current stage only develop mode is enabled.  Only single worker is supported right now.   Test and checkpoint will be conducted every save_freq iterations

You can choose to display or not using ```display flags```

- develop:
```
python main.py --env-name InvertedPendulum-v2 --task develop --save_freq 1000 --display True
```

This project also defines it's own environements under env/ directory.   Train a custom inverted pendulum environment with
```
python main.py --env-name InvertedPendulum-down --task develop --save_freq 1000 --display True
```

This is a much harder task because the pole is initially facing down, so the agent needs to learn to 'swing' the pole up then keep it there.  You can tweek this task by set a proper reward function.

In some case that you want to check if you code runs as you want, you might resort to ```pdb```. Here, I provide a develop mode, which only runs in one thread (easy to debug).


## Experiment results

Experiment result of this fork to be added later

### learning curve

The plot of total reward/episode length in 1000 steps:

- InvertedPendulum-v1

![](asset/InvertedPendulum-v1.a3c.log.png)

In InvertedPendulum-v1, total reward exactly equal to episode length.

- InvertedDoublePendulum-v1

![](asset/InvertedDoublePendulum-v1.a3c.log.png)

**Note that the x axis denote the time in minute**

The above curve is plotted from ```python plot.py --log_path ./logs/a3c/InvertedPendulum-v1.a3c.log```


### video

- InvertedPendulum-v1

<a href="http://www.youtube.com/watch?feature=player_embedded&v=E7QlRIkKuXo
" target="_blank"><img src="http://img.youtube.com/vi/E7QlRIkKuXo/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

- InvertedDoublePendulum-v1

<a href="http://www.youtube.com/watch?feature=player_embedded&v=WNiitHoz8x4
" target="_blank"><img src="http://img.youtube.com/vi/WNiitHoz8x4/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

## Requirements
- gym
- mujoco-py
- pytorch
- matplotlib (optional)
- seaborn (optional)

## TODO
I implement the ShareRMSProp in ```my_optim.py```, but I haven't tried it yet.

## Reference
- [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
