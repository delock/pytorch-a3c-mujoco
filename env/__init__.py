import logging
from gym.envs.registration import register
from env.inverted_pendulum_down import InvertedPendulumDownEnv

logger = logging.getLogger(__name__)

register(
    id="InvertedPendulum-down",
    entry_point="env:InvertedPendulumDownEnv",
    max_episode_steps=500,
    reward_threshold=500.0,
)
