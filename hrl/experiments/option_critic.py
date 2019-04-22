import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp
import redis
import pickle
from tqdm import tqdm
from functools import partial

from hrl.utils import ROOT_DIR
from hrl.frameworks.options.OptionCritic import OptionCritic
from hrl.frameworks.options.learned_options import LearnedOption
from hrl.envs.FourRooms import FourRooms, stochastic_step


class LearningRateScheduler:
    
    def __init__(self, decays: dict):
        for rate, schedule in decays.items():
            setattr(self, rate, schedule)
    
    def time_decay(self, t):
        for rate, schedule in vars(self).items():
            schedule['rate'] = max(schedule['min'], schedule['rate'] * (1. / (1. + schedule['decay'] * t)))


if __name__ == '__main__':
    
    db = redis.StrictRedis(port=6379)
    
    softmax = lambda v: np.exp(v - logsumexp(v))
    
    goals = [(15, 15), (10, 17), (17, 10)]
    
    # Create environment
    env = FourRooms(agent_pos=(1, 1), goal_pos=(15, 15))
    env.max_steps = 1000000
    env.step = partial(stochastic_step, env)
    
    # Create options
    n_options = 4
    dim = (4, 19, 19)
    n_states = np.prod(np.array(dim))
    n_actions = 3
    options = [LearnedOption(n_states, n_actions) for _ in range(n_options)]
    
    agent = OptionCritic(env=env, options=options)
    n_episodes = 5000
    
    # Learning rate schedules
    learning_rates = dict(
        alpha_critic=dict(
            rate=agent.alpha_critic,
            decay=agent.alpha_critic / 10000,
            min=0.01,
        ),
        alpha_theta=dict(
            rate=agent.alpha_theta,
            decay=agent.alpha_theta / 5000,
            min=0.001,
        ),
        alpha_upsilon=dict(
            rate=agent.alpha_upsilon,
            decay=agent.alpha_upsilon / 5000,
            min=0.001,
        )
    )
    scheduler = LearningRateScheduler(learning_rates)
    
    for t in tqdm(range(n_episodes)):
        agent.run_episode(render=False)
        scheduler.time_decay(t)
        for rate, alpha in vars(scheduler).items():
            setattr(agent, rate, alpha['rate'])
        print(f'Q LR: {agent.alpha_critic}, PG LR {agent.alpha_theta}, Term LR {agent.alpha_upsilon},')
        
        # if t == 500:
        #     agent.env._goal_default_pos = goals[1]
        
        # Q = np.reshape(agent.Q, (*dim, n_options))
        # Qu = np.reshape(agent.Q_U, (*dim, n_options, 3))
        
        # term = np.zeros((n_options, *dim))
        # pi = term.copy()
        #
        # for i in range(n_options):
        #     term[i] = np.reshape(agent.options[i].beta.weights, dim)
        #     policy = np.argmax(softmax(agent.options[i].policy.weights), axis=1)
        #     pi[i] = np.reshape(policy, dim)
        #
        # # if i and (i % 10 == 0):
        # db.set('Options-Q-Values', pickle.dumps(Q))
        # # db.set('Options-Qu-Values', pickle.dumps(Qu))
        # db.set('Terminations', pickle.dumps(term))
        # db.set('Policies', pickle.dumps(pi))
