from gym_minigrid.minigrid import MiniGridEnv
import numpy as np
import random
import time
from hrl.utils import randargmax
from plotly import graph_objs as go
from typing import List


class SMDPValueLearning:
    """
    Value learning in Semi-Markov Decision Process via Q-learning.
    
    Does not work well in this setting, since the rooms are larger than in the original experiment
    and thus the probability of stumbling across a goal state, while performing primitive actions only is much smaller.
    Consider a case when the agent is at the hallway state. It can try to make a primitive action
    in the direction of the goal. However, at the next state it can choose to take an option that takes it
    either to the other hallway or back with probability 2/5. As the agent makes progress towards the goal state,
    it's more likely that the option will get activated along the way. Since no intra-option learning is happening,
    the value is not attributed to the states surrounding the goal.
    """
    
    def __init__(self, env: MiniGridEnv, options: List):
        self.env = env
        self.options = options
        
        self.k = 0  # Duration of the current option
        self.starting_state = None  # Initiation time
        self.current_option = None  # Pointer to the current option object
        self.cumulative_reward = 0  # Reward accumulated by following the current option
    
    def reset_option(self):
        self.k = 0
        self.starting_state = None
        self.current_option = None
        self.cumulative_reward = 0
    
    def _choose_optimal_option(self, q_values, state):
        """ filters options according to availability and picks one greedily"""
        
        all_options = q_values[:, state[0], state[1], state[2]]
        available_options = [i for i, o in enumerate(self.options) if o.initiation_set[state] == 1]
        # print('Available options:', [self.options[o].name for o in available_options])
        
        # Get max across all available options
        max_q = all_options[available_options].max()
        
        # Pick at random from the best available options
        valid = set(available_options) & set(np.where(all_options == max_q)[0])
        optimal_option = random.choice(list(valid))
        
        # print('Best option:', self.options[optimal_option].name)
        suboptimal_options = [self.options[i] for i in available_options]
        return optimal_option, suboptimal_options
    
    def choose_option(self, q_values, state, epsilon=0.1):
        if self.current_option is None:
            # Pick the optimal option with probability 1 - epsilon
            optimal_option, suboptimal_options = self._choose_optimal_option(q_values, state)
            if random.random() > epsilon:
                self.current_option = self.options[optimal_option]
            else:
                self.current_option = random.choice(suboptimal_options)
            
            # Remember the state that the option started in
            self.starting_state = state
        
        # Select action according to the policy of our current option
        action = self.current_option.choose_action(state)
        return action
    
    def q_learning(self, q_values: np.array = None, n_visits: np.array = None,
                   epsilon: float = 0.1, step_size: float = 0.25, gamma: float = 0.9, render: bool = False):
        
        # State is number of cells in the grid plus the direction of agent
        # Options consist of primitive {left, right, forward} and multi-step hallway options
        n_options = len(self.options)
        if q_values is None:
            n_visits = np.zeros((n_options, 4, self.env.width, self.env.height))
            q_values = np.zeros((n_options, 4, self.env.width, self.env.height))
        
        self.env.reset()
        self.reset_option()
        self.starting_state = state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done, a = False, None
        
        while not done:
            if render:
                print()
                print(
                    f"State: {state}, Option: {self.current_option.name if self.current_option else None}, Action: {a}")
                self.env.render()
                time.sleep(0.05)
            
            a = self.choose_option(q_values, state, epsilon)
            obs, reward, done, info = self.env.step(a)
            
            # Note: we could infer the state of the agent from obs, but get it directly instead
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))
            
            # Update option state
            self.k += 1
            self.cumulative_reward += gamma ** self.k * reward
            
            # Check for termination condition and update action-values. Notice the discounting factor power.
            if self.current_option.termination_set[state_next] == 1 or done:
                
                option_state = (self.options.index(self.current_option), *self.starting_state)
                
                if step_size is None:
                    n_visits[option_state] += 1
                    alpha = 1 / n_visits[option_state]
                else:
                    alpha = step_size
                
                a_next = randargmax(q_values[:, state_next[0], state_next[1], state_next[2]])
                target = self.cumulative_reward + gamma ** self.k * q_values[(a_next, *state_next)]
                q_values[option_state] += alpha * (target - q_values[option_state])
                
                self.reset_option()
            
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
    
    def __init__(self, env: MiniGridEnv, options: List):
        self.env = env
        self.options = options
        
        self.k = 0  # Duration of the current option
        self.starting_state = None  # Initiation time
        self.current_option = None  # Pointer to the current option object
        self.cumulative_reward = 0  # Reward accumulated by following the current option
    
    def reset_option(self):
        self.k = 0
        self.starting_state = None
        self.current_option = None
        self.cumulative_reward = 0
    
    def choose_option(self, state):
        if self.current_option is None:
            # Pick an option at random
            available_options = [o for o in self.options if o.initiation_set[state] == 1]
            print([o.name for o in available_options])
            self.current_option = random.choice(available_options)
            self.starting_state = state
        
        # Select action according to the policy of our current option
        action = self.last_action = self.current_option.choose_action(state)
        return action
    
    def run_episode(self, R: np.array = None, P: np.array = None, N: np.array = None, gamma: float = 0.9,
                    render: bool = False):
        
        # State is number of cells in the grid plus the direction of agent
        # Options consist of primitive {left, right, forward} and multi-step hallway options
        
        n_options = len(self.options)
        state_space = (4, self.env.width, self.env.height)
        dim = (n_options, *state_space)
        
        if R is None:
            R = np.zeros(dim)
            N = np.zeros(dim)
            P = np.zeros((n_options, *state_space, *state_space))
        
        """
        In this experiment, the rewards
        were selected according to a normal probability distribution with a standard deviation of
        0.1 and a mean that was dierent for each state{action pair. The means were selected
        randomly at the beginning of each run uniformly from the [−1; 0] interval.
        """
        rewards = np.random.normal(np.random.uniform(-1, 0, dim), 0.1 + np.zeros(dim))
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done = False
        
        while not done:
            
            a = self.choose_option(state)
            obs, reward, done, info = self.env.step(a)
            if not done:
                reward = rewards[(a, *state)]
            
            if render:
                print(f"State: {state}, Option: {self.current_option.name}, Action: {a}")
                self.env.render()
                time.sleep(0.05)
            
            # Note: we could infer the state of the agent from obs, but get it directly instead
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))
            
            # Update option state
            self.k += 1
            self.cumulative_reward += gamma ** self.k * reward
            
            # Check for termination condition and update the model
            if self.current_option.termination_set[state_next] == 1:
                # check if last state had the same x,y , to prevent frequent option termination
                
                option_state = (self.options.index(self.current_option), *self.starting_state)
                # Update visitation counter
                N[option_state] += 1
                alpha = (1 / N[option_state])
                
                # Update reward matrix
                R[option_state] += alpha * (self.cumulative_reward - R[option_state])
                
                # Update probability transition matrix
                P[(*option_state, *state_next)] += alpha * (gamma ** self.k)
                P[option_state] -= alpha * P[option_state]
                
                self.reset_option()
            
            state = state_next
        
        return N, R, P
