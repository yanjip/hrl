import pickle
import redis
from tqdm import tqdm
from functools import partial

from hrl.frameworks.options.SMDP import SMDPValueLearning
from hrl.frameworks.options.IntraOption import IntraOptionValueLearning
from hrl.utils import ROOT_DIR

from hrl.envs.FourRooms import FourRooms, stochastic_step
from hrl.frameworks.options.hallway_options import HallwayOption


def create_agent(mode='primitive + options'):
    # Specify the environment
    env = FourRooms(agent_pos=(1, 1), goal_pos=(15, 15))
    env.max_steps = 1000000
    env.step = partial(stochastic_step, env)
    
    # Provide options
    options = {'left', 'right', 'forward'}
    if mode == 'primitive + options':
        options = options | {
            'topleft->botleft',
            'topleft->topright',
            'topright->topleft',
            'topright->botright',
            'botleft->topleft',
            'botleft->botright',
            'botright->botleft',
            'botright->topright'
        }
    options = [HallwayOption(o) for o in options]
    
    # Instantiate agent
    agent = IntraOptionValueLearning(env=env, options=options)
    
    return agent


def learn_values(agent, n_episodes: int, transfer: bool = True, render: bool = False):
    q_values = None
    step_count = list()
    history = dict()
    goals = [(15, 15), (10, 17), (17, 10), (17, 1), (8, 8)]
    
    switch = 1
    for i in tqdm(range(1, n_episodes)):
        
        q_values, steps = agent.q_learning(q_values, render=False)
        
        if transfer and i % 500 == 0:
            history[i] = q_values.copy()
            agent.env._goal_default_pos = goals[switch]
            switch += 1
        
        step_count.append(steps)
        if i % 10 == 0:
            print(sum(step_count) / 10)
            # db.set('Q-Values', pickle.dumps(q_values))
            step_count = []
    
    if transfer:
        with open(f'{ROOT_DIR}/experiment_results/4rooms/transfer_with_options_500steps.pkl', 'wb') as f:
            pickle.dump((history, step_count), f)


if __name__ == "__main__":
    learn_values(agent=create_agent(), n_episodes=1000, transfer=True, render=False)
