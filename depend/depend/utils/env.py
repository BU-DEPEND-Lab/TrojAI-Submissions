import gym_minigrid
import gym


def make_env(env_key, seed=None, render_mode=None):
    print(env_key)
    env = gym.make(env_key)
    env.reset()
    return env