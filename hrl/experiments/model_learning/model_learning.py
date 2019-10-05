import pickle
from typing import List

import gym
import numpy as np
import ray
from gym_minigrid.wrappers import FullyObsWrapper
from plotly import graph_objs as go
import plotly
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


def plot(errors: np.ndarray, curve_names: List[str]):
    # Average across runs
    mean = np.mean(errors, axis=0)
    std = np.std(errors, axis=0)
    upper_bound = mean + std
    lower_bound = mean - std
    
    traces = list()
    for i, curve_name in enumerate(curve_names):
        traces.append(go.Scatter(
            name=f'{curve_name} Lower Bound',
            y=lower_bound[i],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines'))
        
        traces.append(go.Scatter(
            name=f'{curve_name} Mean',
            y=mean[i],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty'))
        
        traces.append(go.Scatter(
            name=f'{curve_name} Upper Bound',
            y=upper_bound[i],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty'))
    
    # Trace order can be important
    # with continuous error bars
    # data = [lower_bound, trace, upper_bound]
    
    layout = go.Layout(
        yaxis=dict(title='Absolute prediction error'),
        showlegend=True)
    
    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig, auto_open=False, filename='model_learning.html')


def learn_model(agent, n_episodes: int, render: bool = False):
    N, R, P = None, None, None
    for _ in tqdm(range(n_episodes)):
        N, R, P = agent.run_episode(R, P, N, render=render)
        yield N, R, P
    
    # Save learned model
    with open(f'{ROOT_DIR}/cache/{agent}_{n_episodes}.pkl', 'wb') as f:
        pickle.dump((N, R, P), f)
    
    return N, R, P


def compute_reward_error(options, true_R, R):
    """ Absolute error in reward prediction over initiation set,
    averaged across all options.
    """
    error = 0
    for i, option in enumerate(options):
        if isinstance(option, PrimitiveOption):
            continue
        abs_rew = np.abs(true_R[i] - R[i])
        error += np.sum(np.multiply(abs_rew, option.initiation_set))
    error /= len(options)
    return error


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

        env = self.unwrapped
        ix = (action, env.agent_dir, *reversed(env.agent_pos))
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
    env = RandomRewards(FullyObsWrapper(
        FourRooms(agent_pos=(1, 1), goal_pos=(0, 0))))
    env.unwrapped.max_steps = 1000000
    # env.step = partial(stochastic_step, env)
    
    # Use hard-coded hallway options
    options = [HallwayOption(o, env.observation_space.shape[::-1]) for o in
               sorted(HallwayOption.hallway_options)]
    options += [PrimitiveOption(o, env.observation_space.shape[::-1]) for o in
                sorted(PrimitiveOption.primitive_options)]


    @ray.remote
    def single_run(env, options, seed, record_every: int = 1000):
    
        # Obtain true models for options
        agent = SMDPModelLearning(env=env, options=options)
        env_name = agent.env.unwrapped.__class__.__name__
        pickler = cache(f'{ROOT_DIR}/cache/{env_name}/true_option_models')
        _, true_R, true_P = pickler(agent.get_true_models)(seed=seed)
    
        # Run planning agents and record errors in the models
        reward_errors = list()
        for agent in (IntraOptionModelLearning, SMDPModelLearning):
            agent = agent(env=env, options=options)
            errors = list()
            np.random.seed(seed)
            for step, (N, R, P) in enumerate(agent.run_episode(render=False)):
                if step % record_every == 0:
                    error = compute_reward_error(options, true_R, R)
                    errors.append(error)
            reward_errors.append(errors)
    
        return reward_errors


    runs = 8
    ray.init(local_mode=False)
    errors = ray.get([single_run.remote(env, options, i) for i in range(runs)])
    plot(errors, curve_names=['IntraOption', 'SMDP'])
    
    # traces.append(
    #     go.Scatter(
    #         mode='lines',
    #         y=errors[i, j],
    #         name=f'SMDP',
    #     )
    # )
