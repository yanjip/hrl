import os

import gym
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines import DQN
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy

from hrl.experiments import EXPERIMENT_DIR


def callback(_locals, _globals):
    n_steps = _locals['_']
    if n_steps and (n_steps % 1000 == 0):
        print(n_steps)
        print(_locals['episode_successes'])
        # env.render()
        # time.sleep(0.2)
    
    n_steps += 1
    # Returning False will stop training early
    return True


# Create log dir
log_dir = f"{EXPERIMENT_DIR}/sb/gym"
os.makedirs(log_dir, exist_ok=True)

# Create environment
env_name = 'MiniGrid-FourRooms-v1'
env = FullyObsWrapper(ImgObsWrapper(gym.make(env_name)))
env.max_steps = 100000
# env.step = partial(stochastic_step, env)
env = DummyVecEnv([lambda: env])

# Train a model
model = DQN(
    policy=MlpPolicy,
    env=env,
    tensorboard_log=f"{EXPERIMENT_DIR}/sb/tensorboard/{env_name}"
)
model.learn(total_timesteps=10000000, callback=callback)
