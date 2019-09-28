import time
from random import random, choice
from typing import List, Tuple

import numpy as np
from gym_minigrid.minigrid import MiniGridEnv
from plotly import graph_objs as go
from tqdm import tqdm

from hrl.frameworks.options.option import Option
from hrl.project_logger import ProjectLogger
from hrl.utils import randargmax

Options = List[Option]


class ExecutingOption(Option):
    
    def __init__(self,
                 k: int = 0,
                 starting_state=None,
                 cumulant_value=0,
                 *args, **kwargs):
        """ An option object that is currently being executed by the agent
        
        :param k: duration of the option so far
        :param starting_state: state in which the option was initiated
        :param cumulant_value: accumulated signal so far
        """
        super().__init__(*args, **kwargs)
        self.k = k
        self.starting_state = starting_state
        self.cumulant_value = cumulant_value
    
    def reset(self):
        self.k = 0
        self.starting_state = None
        self.cumulant_value = 0


class SMDPValueLearning:
    """
    Algorithms for finding an optimal policy over a set of options
    in Semi-Markov Decision Process. Treats each option as an indivisible unit.
    
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
                 options: List[Option],
                 loglevel: int = 20):
        self.env = env
        self.options = options
        self.logger = ProjectLogger(level=loglevel, printing=False)

    def _choose_optimal_option(self,
                               q_values,
                               state) -> Tuple[Option, Options]:
        """ Filters options according to availability and picks one greedily """
    
        # TODO(Vlad): unroll state tuple!
        all_options = q_values[:, state[0], state[1], state[2]]
        options = filter(lambda i, o: o.initiation_set[state] == 1,
                         enumerate(self.options))
        option_indices = [o[0] for o in options]
        self.logger.debug('Available options:', [str(o[0]) for o in options])
        
        # Get max across all available options
        max_q = all_options[option_indices].max()
        
        # Pick at random from the best available options
        valid = set(option_indices) & set(np.where(all_options == max_q)[0])
        optimal_option = self.options[random.choice(list(valid))]
        self.logger.debug(f'Best option: {optimal_option}')
    
        suboptimal_options = [self.options[i] for i, o in options]
        return optimal_option, suboptimal_options

    def choose_option(self,
                      q_values,
                      state,
                      ε: float = 0.1) -> ExecutingOption:
        """ Picks the optimal option with probability 1 - ε """
        # TODO: Generalize to other choosing methods
    
        optimal, suboptimal = self._choose_optimal_option(q_values, state)
        next_option = optimal if random() > ε else choice(suboptimal)
    
        return ExecutingOption(
            starting_state=state,
            initiation_set=next_option.initiation_set,
            termination_function=next_option.termination_function,
            policy=next_option.policy,
            name=str(optimal)
        )

    def q_learning(self,
                   q_values: np.ndarray = None,
                   n_visits: np.ndarray = None,
                   ε: float = 0.1,
                   α: float = 0.25,
                   γ: float = 0.9,
                   render: bool = False):
        
        n_options = len(self.options)
        if q_values is None:
            state_space_dim = (4, self.env.width, self.env.height)
            n_visits = np.zeros((n_options, *state_space_dim))
            q_values = np.zeros((n_options, *state_space_dim))
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done, a, executing_option = False, None, None
        
        while not done:
            if render:
                self.logger.debug(f"\nState: {state}, "
                                  f"Option: {executing_option}, "
                                  f"Action: {a}")
                self.env.render()
                time.sleep(0.05)

            if executing_option is None:
                executing_option = self.choose_option(q_values, state, ε)

            # Select action according to the policy of the current option
            a = executing_option.policy(state)
            obs, reward, done, info = self.env.step(a)

            # Note: we could infer the state of the agent from obs,
            #  but get it directly instead
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))

            # Update option
            executing_option.k += 1
            executing_option.cumulant_value += γ ** executing_option.k * reward

            # Check for termination condition and update action-values.
            # Notice the discounting factor power.
            if executing_option.termination_function(state_next) == 1 or done:
    
                option_state = (
                    self.options.index(executing_option),
                    *executing_option.starting_state)
    
                if α is None:
                    n_visits[option_state] += 1
                    alpha = 1 / n_visits[option_state]
                else:
                    alpha = α
    
                a = randargmax(
                    q_values[:, state_next[0], state_next[1], state_next[2]])
                target = executing_option.cumulant_value\
                         + γ ** executing_option.k * q_values[(a, *state_next)]
                q_values[option_state] += alpha * (
                    target - q_values[option_state])
    
                executing_option = None
            
            state = state_next
        
        return q_values, self.env.step_count
    
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
                 options: List[Option],
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
        option = choice(options)
        return ExecutingOption(
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
        
        n_options = len(self.options)
        state_space_dim = (4, self.env.width, self.env.height)
        dim = (n_options, *state_space_dim)
        
        if R is None:
            R = np.zeros(dim)
            N = np.zeros(dim)
            P = np.zeros((n_options, *state_space_dim, *state_space_dim))
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done = False
        executing_option = None

        pbar = tqdm(position=0)
        while not done:
            pbar.update(1)
            if executing_option is None:
                executing_option = self.choose_option(state)
    
            a = executing_option.policy(state)
            obs, reward, done, info = self.env.step(a)
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))
            
            if render:
                action_name = list(self.env.actions)[a].name
                self.logger.debug(f"State: {state}, "
                                  f"Option: {executing_option}, "
                                  f"Action: {action_name}, "
                                  f"Next State: {state_next}")
                self.env.render()
                time.sleep(0.05)
    
            executing_option.k += 1
            executing_option.cumulant_value += γ ** executing_option.k * reward
            
            # Check for termination condition and update the model
            if executing_option.termination_function(state_next) == 1:
                option_state = (self.option_idx_dict[executing_option.name],
                                *executing_option.starting_state)
                # Update visitation counter
                N[option_state] += 1
                α = 1 / N[option_state]
                
                # Update reward matrix
                R[option_state] += α * (
                    executing_option.cumulant_value - R[option_state])
                
                # Update probability transition matrix
                P[(*option_state, *state_next)] += α * (γ ** executing_option.k)
                P[option_state] -= α * P[option_state]
        
                executing_option = None
            
            state = state_next
            yield N, R, P
        
        return N, R, P

    def get_true_models(self, seed, γ: float = 1):
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
            for state, _ in np.ndenumerate(option.initiation_set):
                option = ExecutingOption(
                    initiation_set=option.initiation_set,
                    termination_function=option.termination_function,
                    policy=option.policy,
                    name=str(option)
                )
                starting_state = state
                env = self.env.unwrapped
                env.agent_dir, env.agent_pos = state[0], tuple(
                    reversed(state[1:]))
                while not option.termination_function(state):
                    a = option.policy(state)
                    obs, reward, done, info = self.env.step(a)
                    env = self.env.unwrapped
                    state = (env.agent_dir, *reversed(env.agent_pos))
                    option.k += 1
                    option.cumulant_value += γ ** option.k * reward
            
                option_state = (option_i, *starting_state)
                R[option_state] = option.cumulant_value
                P[(*option_state, *state)] = γ ** option.k
    
        return N, R, P
