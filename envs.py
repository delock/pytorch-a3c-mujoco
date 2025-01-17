import gym
import numpy as np
from gym.spaces.box import Box
import pdb

# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    return env

# process each frame
def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 42, 42])
    return frame
