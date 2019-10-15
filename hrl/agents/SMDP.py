import random
import time
from typing import List

import numpy as np
from gym_minigrid.minigrid import MiniGridEnv
from plotly import graph_objs as go
from tqdm import tqdm

from hrl.frameworks.options.option import Option, MarkovOption
from hrl.frameworks.options.policies import PolicyOverOptions
from hrl.project_logger import ProjectLogger
from hrl.utils import randargmax

Options = List[Option]


class SMDPValueLearning:
    """ Algorithms for finding an optimal policy over a set of options in SMDP.
    
    Treats each option as an indivisible unit.
    Does not work well in this setting, since the rooms are larger than in the original experiment
    and thus the probability of stumbling across a goal state, while performing primitive actions only is much smaller.
    Consider a case when the agent is at the hallway state. It can try to make a primitive action
    in the direction of the goal. However, at the next state it can choose to take an option that takes it
    either to the other hallway or back with probability 2/5. As the agent makes progress towards the goal state,
    it's more likely that the option will get activated along the way. Since no intra-option learning is happening,
    the value is not attributed to the states surrounding the goal.
    """
    
    def __init__(self,
                 env: MiniGridEnv,
                 options: Options,
                 policy: PolicyOverOptions,
                 loglevel: int = 20):
        self.env = env
        self.options = options
        self.option_names_dict = {o.name: o for o in self.options}
        self.option_idx_dict = {name: i for i, name in
                                enumerate(self.option_names_dict)}
        
        self._policy = policy
        self.logger = ProjectLogger(level=loglevel, printing=False)
    
    def policy(self, state, *args, **kwargs):
        option = self._policy(state, *args, **kwargs)
        return MarkovOption(
            starting_state=state,
            initiation_set=option.initiation_set,
            termination_function=option.termination,
            policy=option.target_policy,
            name=str(option)
        )
    
    def q_learning(self,
                   n_episodes: int,
                   γ: float = 0.9,
                   Q: np.ndarray = None,
                   N: np.ndarray = None,
                   α: float = None,
                   render: bool = False):
        
        env = self.env.unwrapped
        n_options = len(self.options)
        state_space_dim = (4, env.width, env.height)
        dim = (n_options, *state_space_dim)
        
        if Q is None:
            N = np.zeros(dim)
            Q = np.zeros(dim)
        
        for episode in range(n_episodes):
            
            self.env.reset()
            state = (env.agent_dir, *reversed(env.agent_pos))
            executing_option = self.policy(Q, state)
            done = False
            
            while not done:
                
                # Step through environment
                a = executing_option.policy(state)
                obs, reward, done, info = self.env.step(a)
                # TODO: infer the state of the agent from obs, i.e. make it POMDP
                s_next = (env.agent_dir, *reversed(env.agent_pos))
                
                if render:
                    action_name = list(env.actions)[a].name
                    self.logger.debug(f"State: {state}, "
                                      f"Option: {executing_option}, "
                                      f"Action: {action_name}, "
                                      f"Next State: {s_next}")
                    self.env.render()
                    time.sleep(0.05)
                
                # Update option
                executing_option.k += 1
                executing_option.cumulant += γ ** executing_option.k * reward
                
                # Check for termination condition and update action-values
                if executing_option.termination_function(s_next) == 1 or done:
                    
                    start_state = (self.option_idx_dict[executing_option.name],
                                   *executing_option.starting_state)
                    
                    # Determine the step-size
                    if α is None:
                        N[start_state] += 1
                        alpha = 1 / N[start_state]
                    else:
                        alpha = α
                    
                    # Update Q in the direction of the optimal action
                    r = executing_option.cumulant
                    k = executing_option.k
                    o = randargmax(Q[(slice(None), *s_next)])
                    target = r + γ ** k * Q[(o, *s_next)]
                    Q[start_state] += alpha * (target - Q[start_state])
                    
                    # Choose the next option
                    executing_option = self.policy(Q, s_next)
                
                # Reset the state
                state = s_next
            yield Q, self.env.step_count
        
        return Q, self.env.step_count
    
    @staticmethod
    def plot_episode_duration(steps_per_episode):
        traces = list()
        for option_set, steps_per_episode in steps_per_episode.items():
            traces.append(
                go.Scatter(
                    mode='lines',
                    y=steps_per_episode,
                    name=option_set,
                )
            )
        
        layout = dict(
            height=700,
            showlegend=True,
            xaxis=dict(
                title='Episodes',
            ),
            yaxis=dict(
                title='Steps per episode',
            )
        )
        return {'data': traces, 'layout': layout}


class SMDPModelLearning:
    """ Model learning in Semi-Markov Decision Process via MC sampling """
    
    def __init__(self,
                 env: MiniGridEnv,
                 options: Options,
                 loglevel: int = 10):
        self.env = env
        self.options = options
        self.option_names_dict = {o.name: o for o in self.options}
        self.option_idx_dict = {name: i for i, name in
                                enumerate(self.option_names_dict)}
        self.logger = ProjectLogger(level=loglevel, printing=True)
    
    def __str__(self):
        return 'SMDPModelLearning'
    
    def choose_option(self, state):
        """ Picks an option at random """
        options = [o for o in self.options if o.initiation_set[state] == 1]
        option = random.choice(options)
        return MarkovOption(
            starting_state=state,
            initiation_set=option.initiation_set,
            termination_function=option.termination_function,
            policy=option.policy,
            name=str(option)
        )
    
    def run_episode(self,
                    N: np.ndarray = None,
                    R: np.ndarray = None,
                    P: np.ndarray = None,
                    γ: float = 0.9,
                    render: bool = False):
        
        env = self.env.unwrapped
        n_options = len(self.options)
        state_space_dim = (4, env.width, env.height)
        dim = (n_options, *state_space_dim)
        
        if R is None:
            R = np.zeros(dim)
            N = np.zeros(dim)
            P = np.zeros((n_options, *state_space_dim, *state_space_dim))
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(env.agent_pos))
        done = False
        executing_option = None
        
        while not done:
            
            if executing_option is None:
                executing_option = self.choose_option(state)
            
            a = executing_option.policy(state)
            obs, reward, done, info = self.env.step(a)
            state_next = (env.agent_dir, *reversed(env.agent_pos))
            
            if render:
                action_name = list(env.actions)[a].name
                self.logger.debug(f"State: {state}, "
                                  f"Option: {executing_option}, "
                                  f"Action: {action_name}, "
                                  f"Next State: {state_next}")
                self.env.render()
                time.sleep(0.05)
            
            executing_option.k += 1
            executing_option.cumulant += γ ** executing_option.k * reward
            
            # Check for termination condition and update the model
            if executing_option.termination_function(state_next) == 1:
                option_state = (self.option_idx_dict[executing_option.name],
                                *executing_option.starting_state)
                # Update visitation counter
                N[option_state] += 1
                α = 1 / N[option_state]
                
                # Update reward matrix
                R[option_state] += α * (
                    executing_option.cumulant - R[option_state])
                
                # Update probability transition matrix
                P[(*option_state, *state_next)] += α * (γ ** executing_option.k)
                P[option_state] -= α * P[option_state]
                
                executing_option = None
            
            state = state_next
            yield N, R, P
        
        return N, R, P
    
    def get_true_models(self, seed: int = 1337, γ: float = 0.9):
        """ Learn true dynamics (P) and reward (R) models by unrolling each
        option for each state in its initiation set until termination
        # TODO: need to call multiple times if the environment is stochastic?
        """
        
        np.random.seed(seed)
        
        n_options = len(self.options)
        state_space_dim = (4, self.env.width, self.env.height)
        dim = (n_options, *state_space_dim)
        
        R = np.zeros(dim)
        N = np.zeros(dim)
        P = np.zeros((n_options, *state_space_dim, *state_space_dim))
        
        self.env.reset()
        
        for option_i, option in tqdm(enumerate(self.options)):
            for state, active in np.ndenumerate(option.initiation_set):
                
                # Checks if the state is in initiation set
                if not active:
                    continue
                
                env = self.env.unwrapped
                env.agent_dir, env.agent_pos = state[0], tuple(
                    reversed(state[1:]))
                cell = self.env.grid.get(*env.agent_pos)
                
                # Check if the state is valid for the agent to be in
                if not (cell is None or cell.can_overlap()):
                    continue
                
                # Activate an option and run until termination
                option = MarkovOption(
                    starting_state=state,
                    initiation_set=option._initiation_set,
                    termination_set=option._termination_set,
                    policy=option._policy,
                    name=str(option)
                )
                while True:
                    a = option.policy(state)
                    obs, reward, done, info = self.env.step(a)
                    env = self.env.unwrapped
                    state_next = (env.agent_dir, *reversed(env.agent_pos))
                    self.logger.debug(f"State: {state}, "
                                      f"Option: {option}, "
                                      f"Action: {a}, "
                                      f"Next State: {state_next}")
                    state = state_next
                    option.k += 1
                    option.cumulant += γ ** option.k * reward
                    if option.termination_function(state):
                        break
                
                # Update option models
                option_state = (option_i, *option.starting_state)
                R[option_state] = option.cumulant
                P[(*option_state, *state)] = γ ** option.k
        
        return N, R, P


class SMDPPlanning:
    """ Estimates value function given learned models R and P """
    
    def __init__(self,
                 env: MiniGridEnv,
                 R: np.ndarray,
                 P: np.ndarray,
                 loglevel: int = 10):
        self.env = env
        
        option_dim, self.state_space_dim = R.shape[0], R.shape[1:]
        state_space_flat = np.prod(self.state_space_dim)
        self.R = R.reshape((option_dim, state_space_flat))
        self.P = P.reshape((option_dim, state_space_flat, state_space_flat))
        self.V = np.zeros(state_space_flat)
        
        self.logger = ProjectLogger(level=loglevel, printing=True)
    
    def svi(self, θ: float = 1e-9):
        """ Iterative Policy Evaluation using Synchronous Value Iteration.
        
        Estimates V by acting greedy wrt to V_hat (current estimate of V)
        """
        δ = float('inf')
        
        while δ > θ:
            v_old = self.V
            self.V = (self.R + np.dot(self.P, self.V)).max(axis=0)
            δ = np.sum(np.abs(self.V - v_old))
            self.logger.debug(f'State-value delta: {δ}')
            yield self.V.reshape(self.state_space_dim)
        
        return self.V.reshape(self.state_space_dim)
