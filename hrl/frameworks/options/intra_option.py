import random
import time
from typing import List

import numpy as np
from gym_minigrid.minigrid import MiniGridEnv

from hrl.frameworks.options.option import Option
from hrl.project_logger import ProjectLogger
from hrl.utils import randargmax


class IntraOptionModelLearning:
    
    def __init__(self, env: MiniGridEnv, options: List[Option],
                 loglevel: int = 20):
        self.env = env
        self.options = options
        self.option_names_dict = {o.name: o for o in self.options}
        self.option_idx_dict = {name: i for i, name in
                                enumerate(self.option_names_dict)}
        self.logger = ProjectLogger(level=loglevel, printing=False)
    
    def __str__(self):
        return 'IntraOptionModelLearning'
    
    def choose_option(self, state):
        """ Picks an option at random """
        options = [o for o in self.options if o.initiation_set[state] == 1]
        return random.choice(options)
    
    def run_episode(self,
                    R: np.ndarray = None,
                    P: np.ndarray = None,
                    N: np.ndarray = None,
                    γ: float = 0.9,
                    α: float = 1 / 4,
                    render: bool = False):
        
        n_options = len(self.options)
        state_space = (4, self.env.width, self.env.height)
        dim = (n_options, *state_space)
        
        if R is None:
            R = np.zeros(dim)
            N = np.zeros(dim)
            P = np.zeros((n_options, *state_space, *state_space))
        
        rewards = np.random.normal(
            loc=np.random.uniform(-1, 0, dim),
            scale=0.1 + np.zeros(dim)
        )
        self.env.rewards = rewards
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done = False
        executing_option = None
        
        while not done:
    
            if executing_option is None:
                executing_option = self.choose_option(state)
    
            a = executing_option.policy(state)
            obs, reward, done, info = self.env.step(a)
            # TODO: make the environment aware of options?
            reward = rewards[
                (self.option_idx_dict[executing_option.name], *state)]
            s_next = (self.env.unwrapped.agent_dir,
                      *reversed(self.env.unwrapped.agent_pos))
            
            if render:
                action_name = list(self.env.actions)[a].name
                self.logger.debug(f"State: {state}, "
                                  f"Option: {executing_option}, "
                                  f"Action: {action_name}, "
                                  f"Next State: {s_next}")
                self.env.render()
                time.sleep(0.05)
            
            # Update model for every option consistent with last action taken
            for option in self.options:
                if option.policy(state) != a:
                    continue
    
                o = self.option_idx_dict[executing_option.name]
                option_state = (o, *state)
                
                # Update visitation counter
                if α is None:
                    N[option_state] += 1
                    alpha = 1 / N[option_state]
                else:
                    alpha = α
                
                # Update reward matrix
                β = option.termination_function(s_next)
                target = reward + γ * (1 - β) * R[(o, *s_next)]
                R[option_state] += alpha * (target - R[option_state])
                
                # Update probability transition matrix
                target = γ * (1 - β) * P[o, :, :, :, s_next[0], s_next[1],
                                       s_next[2]]
                P[option_state] += alpha * (target - P[option_state])
                P[(o, *state, *s_next)] += alpha * γ * β
    
            if executing_option.termination_function(s_next) == 1:
                executing_option = None
    
            state = s_next
        
        return N, R, P


class IntraOptionValueLearning:
    """
        We turn now to the intra-option learning of option values and thus of optimal policies
        over options. If the options are semi-Markov, then again the SMDP methods described in
        Section 5.2 are probably the only feasible methods; a semi-Markov option must be completed
        before it can be evaluated in any way. But if the options are Markov and we are willing to
        look inside them, then we can consider intra-option methods. Just as in the case of model
        learning, intra-option methods for value learning are potentially more efficient than SMDP
        methods because they extract more training examples from the same experience.

        Note: the following approach allows for option execution, while learning about the options
            that are consistent with the primitive actions taken. In the original paper, only primitive
            actions were taken in order to learn the option values.
    """
    
    def __init__(self, env: MiniGridEnv, options: List):
        self.env = env
        self.options = options
        self.current_option = None
        self.last_action = None
    
    def _choose_optimal_option_(self, q_values, state):
        """ filters options according to availability and picks one greedily"""
        
        all_options = q_values[:, state[0], state[1], state[2]]
        available_options = [i for i, o in enumerate(self.options) if
                             o.initiation_set[state] == 1]
        # print('Available options:', [self.options[o].name for o in available_options])
        
        # Get max across all available options
        max_q = all_options[available_options].max()
        
        # Pick at random from the best available options
        valid = set(available_options) & set(np.where(all_options == max_q)[0])
        optimal_option = random.choice(list(valid))
        
        # print('Best option:', self.options[optimal_option].name)
        suboptimal_options = [self.options[i] for i in available_options if
                              i != optimal_option]
        return optimal_option, suboptimal_options
    
    def _choose_optimal_option(self, q_values, state):
        """ filters options according to availability and picks one greedily"""
        
        all_options = q_values[:, state[0], state[1], state[2]]
        available_options = [i for i, o in enumerate(self.options) if
                             o.initiation_set[state] == 1]
        # print('Available options:', [self.options[o].name for o in available_options])
        
        # Get max across all available options
        max_q = all_options[available_options].max()
        
        # Pick at random from the best available options
        valid = set(available_options) & set(np.where(all_options == max_q)[0])
        if self.current_option and q_values[
            (self.options.index(self.current_option), *state)] == max_q:
            optimal_option = self.options.index(self.current_option)
        else:
            optimal_option = random.choice(list(valid))
        
        # print('Best option:', self.options[optimal_option].name)
        suboptimal_options = [self.options[i] for i in available_options if
                              i != optimal_option]
        return optimal_option, suboptimal_options
    
    def choose_option_(self, q_values, state, epsilon=0.1):
        if self.current_option is None:
            # Pick the optimal option with probability 1 - epsilon
            optimal_option, suboptimal_options = self._choose_optimal_option(
                q_values, state)
            if random.random() > epsilon:
                self.current_option = self.options[optimal_option]
            else:
                self.current_option = random.choice(suboptimal_options)
        
        # Select action according to the policy of our current option
        action = self.last_action = self.current_option.choose_action(state)
        return action
    
    def choose_option(self, q_values, state, epsilon=0.1):
        # Pick the optimal option with probability 1 - epsilon
        optimal_option, suboptimal_options = self._choose_optimal_option(
            q_values, state)
        if random.random() > epsilon:
            self.current_option = self.options[optimal_option]
        else:
            self.current_option = random.choice(suboptimal_options)
        
        # Select action according to the policy of our current option
        action = self.last_action = self.current_option.choose_action(state)
        return action
    
    def q_learning(self, q_values: np.array = None,
                   epsilon: float = 0.1, alpha: float = 1 / 4,
                   γ: float = 0.9, render: bool = False):
        
        # State is number of cells in the grid plus the direction of agent
        # Options consist of primitive {left, right, forward} and multi-step hallway options
        if q_values is None:
            state_space_dim = (4, self.env.width, self.env.height)
            q_values = np.zeros((len(self.options), *state_space_dim))
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done, a = False, None
        
        while not done:
            if render:
                self.env.render()
                time.sleep(0.05)
            
            a = self.choose_option(q_values, state, epsilon)
            obs, reward, done, info = self.env.step(a)
            
            # Note: we could infer the state of the agent from obs, but get it directly instead
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))
            o_next = randargmax(
                q_values[:, state_next[0], state_next[1], state_next[2]])
            
            # Update action-values for every option consistent with last action taken
            for option in self.options:
                if option.choose_action(state) != self.last_action:
                    continue
                
                # if done:
                #     print(f"State: {state}, Option: {option.name if self.current_option else None}, Action: {a}")
                #     print(state_next, o_next)
                
                option_index = self.options.index(option)
                option_state = (option_index, *state)

                continuation_value = (1 - option.termination_set[state_next]) *\
                                     q_values[(option_index, *state_next)]
                termination_value = option.termination_set[state_next] *\
                                    q_values[(o_next, *state_next)]
                U = continuation_value + termination_value

                target = reward + γ * U
                q_values[option_state] += alpha * (
                    target - q_values[option_state])
            
            # Terminate the option
            if self.current_option.termination_set[state_next] == 1 or done:
                self.current_option = None
            
            state = state_next
        
        return q_values, self.env.step_count
