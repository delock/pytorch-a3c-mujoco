import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os


# An experimental pendulum that pole start dangled down
class InvertedPendulumDownEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "inverted_pendulum_down.xml"), 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        '''
        if np.abs(ob[1]) < 0.2:
            reward = 1
        else:
            reward = 0.2/np.abs(ob[1])
        '''
        two_pi = 2*3.1415926536
        val = ob[1]/two_pi
        val = val - int(val) + 1
        val = val - int(val)
        reward = np.abs(2*(0.5-val))
        #print ("reward = {}, ob[1]={}".format(reward, ob[1]))
        #reward = min(np.abs(ob[1]-3.14325), 1/(np.abs(ob[2])+np.abs(ob[3])+0.001))
        #done = True if np.abs(ob[1]) < 0.2 and np.abs(ob[2]) + np.abs(ob[3]) < 0.1 else False
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
