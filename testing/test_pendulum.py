#Importing OpenAI gym package and MuJoCo engine
import gym
import numpy as np
import mujoco_py
import matplotlib.pyplot as plt
import env

#Setting MountainCar-v0 as the environment
env = gym.make('InvertedPendulum-down')
#Sets an initial state
env.reset()
print (env.action_space)
# Rendering our instance 300 times
i = 0
while True:
  #renders the environment
  env.render()
  #Takes a random action from its action space 
  # aka the number of unique actions an agent can perform
  action = env.action_space.sample()
  ob, reward, done, _ = env.step([-5])
  if i == 0:
    print (action)
    print ("ob = {}, reward = {}, done = {}".format(ob, reward, done))
    i += 1
env.close()
