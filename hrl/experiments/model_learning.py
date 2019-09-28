import pickle

import gym
import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper
from plotly import graph_objs as go
from tqdm import tqdm

from hrl.envs.four_rooms import FourRooms
from hrl.frameworks.options.SMDP import SMDPModelLearning
from hrl.frameworks.options.hard_coded_options import HallwayOption, PrimitiveOption
from hrl.frameworks.options.intra_option import IntraOptionModelLearning
from hrl.utils import ROOT_DIR, cache

RewardModel = DynamicsModel = np.ndarray


def plot_model_prediction_errors(errors, error_type):
    traces = list()
    for i, agent in enumerate({'SMDP', 'Intra'}):
        for j, metric in enumerate({'Mean Error', 'Max Error'}):
            traces.append(
                go.Scatter(
                    mode='lines',
                    y=errors[i, j],
                    name=f'{i} {j}',
                )
            )
    
    layout = go.Layout(
        height=700,
        showlegend=True,
        title=error_type,
        xaxis=dict(
            title='Episodes',
        ),
        yaxis=dict(
            title='Absolute error',
        )
    )
    return {'data': traces, 'layout': layout}


def learn_model(agent, n_episodes: int, render: bool = False):
    N, R, P = None, None, None
    for _ in tqdm(range(n_episodes)):
        N, R, P = agent.run_episode(R, P, N, render=render)
        yield N, R, P
    
    # Save learned model
    with open(f'{ROOT_DIR}/cache/{agent}_{n_episodes}.pkl', 'wb') as f:
        pickle.dump((N, R, P), f)
    
    return N, R, P


class RandomRewards(gym.core.Wrapper):
    """ In this experiment, the rewards were selected according to
    normal probability distribution with a standard deviation of 0.1
    and a mean that was different for each state-action pair.
    The means were selected randomly at the beginning of each run uniformly
    from the [−1; 0] interval.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.rewards = None
        self.reset()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        ix = (action, self.env.agent_dir, *reversed(self.env.agent_pos))
        reward = self.rewards[ix]
        
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        dim = (3, 4, 19, 19)
        # self.rewards = -1 * np.ones(shape=dim)
        self.rewards = np.random.normal(
            loc=np.random.uniform(-1, 0, dim),
            scale=0.1 + np.zeros(dim)
        )
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    
    # Specify the environment
    env = FullyObsWrapper(
        RandomRewards(FourRooms(agent_pos=(1, 1), goal_pos=(0, 0))))
    env.unwrapped.max_steps = 1000000
    # env.step = partial(stochastic_step, env)
    
    # Use hard-coded hallway options
    options = [HallwayOption(o, env.observation_space.shape[::-1]) for o in
               HallwayOption.hallway_options]
    options += [PrimitiveOption(o, env.observation_space.shape[::-1]) for o in
                PrimitiveOption.primitive_options]
    
    n_runs = 30
    traces = list()
    
    # Run SMDP agent
    agent = SMDPModelLearning(env=env, options=options)
    reward_errors, model_errors = list(), list()
    
    N, R, P = None, None, None
    for i in tqdm(range(n_runs)):
        
        env_name = agent.env.unwrapped.__class__.__name__
        pickler = cache(f'{ROOT_DIR}/cache/{env_name}/true_option_models')
        N, true_R, true_P = pickler(agent.get_true_models)(seed=i)
        continue
        # Run the model learning agent
        errors = list()
        step = 0
        np.random.seed(i)
        for N, R, P in agent.run_episode(render=False):
            step += 1
            if step % 1000 != 0:
                continue
            
            # Absolute error in prediction over initiation set
            # averaged across all options
            error = 0
            for i, option in enumerate(agent.options):
                if isinstance(option, PrimitiveOption):
                    continue
                
                abs_rew = np.abs(agent.env.rewards[i] - R[i])
                error += np.sum(np.multiply(abs_rew, option.initiation_set))
            error /= len(options)
            print(error)
            errors.append(error)
    
    # traces.append(
    #     go.Scatter(
    #         mode='lines',
    #         y=errors[i, j],
    #         name=f'SMDP',
    #     )
    # )
    
    # Run intra-option agent
    agent = IntraOptionModelLearning(env=env, options=options)
    N, R, P = learn_model(agent=agent, n_episodes=n_episodes, render=False)
