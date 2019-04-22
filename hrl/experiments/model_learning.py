import pickle
from tqdm import tqdm
from functools import partial

from hrl.utils import ROOT_DIR
from hrl.frameworks.options.SMDP import SMDPModelLearning
from hrl.frameworks.options.IntraOption import IntraOptionModelLearning

from hrl.envs.FourRooms import FourRooms, stochastic_step
from hrl.frameworks.options.hallway_options import HallwayOption

from plotly import graph_objs as go


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


def create_agent():
    # Specify the environment
    env = FourRooms(agent_pos=(1, 1), goal_pos=(15, 15))
    env.max_steps = 1000000
    env.step = partial(stochastic_step, env)
    
    # Provide options
    options = {
        'left', 'right', 'forward',
        'topleft->botleft', 'topleft->topright',
        'topright->topleft', 'topright->botright',
        'botleft->topleft', 'botleft->botright',
        'botright->botleft', 'botright->topright'
    }
    options = [HallwayOption(o) for o in options]
    
    # Instantiate agent
    agent = SMDPModelLearning(env=env, options=options)
    
    return agent


def learn_model(agent, n_episodes: int, render: bool = False):
    N = None  # Visitation count
    R = None  # Reward model
    P = None  # Transition probability model
    for _ in tqdm(range(n_episodes)):
        N, R, P = agent.run_episode(N, R, P, render=render)
    
    # Save learned model
    with open(f'{ROOT_DIR}/cache/option_models.pkl', 'wb') as f:
        pickle.dump((N, R, P), f)


if __name__ == "__main__":
    learn_model(agent=create_agent(), n_episodes=2500, render=True)
