import gym_minigrid
import gym

from depend.utils.world import RandomLavaWorldEnv
from depend.utils.wrappers import ObsEnvWrapper, TensorWrapper
 
 
def make_env(env_key, seed=None, render_mode=None, wrapper = 'ImgObsWrapper'):
    print(env_key)
    env = TensorWrapper(ObsEnvWrapper(RandomLavaWorldEnv(mode='simple', grid_size=9), mode='simple'))
    env.reset()
    return env


