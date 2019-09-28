import os
import pickle
import random
import time
from enum import IntEnum

import gym
import numpy as np
from gym_minigrid.minigrid import Wall, Goal
from tqdm import tqdm

from hrl.frameworks.options.option import Option
from hrl.utils import randargmax, ROOT_DIR


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    
    # Done completing task
    done = 3


class PrimitiveOption(Option):
    """ Primitive options are defined everywhere and can terminate anywhere """
    primitive_options = {'left', 'right', 'forward'}
    
    def __init__(self, name: str, state_space_dim: tuple):
        # FIXME(Vlad)
        state_space_dim = (4, 19, 19)
        
        initiation_set = np.ones(state_space_dim)
        termination_set = np.ones(state_space_dim)
        π = np.zeros(state_space_dim) + getattr(Actions, name)
        super().__init__(initiation_set=initiation_set,
                         termination_function=lambda s: termination_set[s],
                         policy=lambda s: int(π[s]),
                         name=name)


class HallwayOption(Option):
    """ Hard-coded options for navigating between rooms in four-rooms grid """
    # TODO(Vlad): make compatible with other rooms environments
    
    hallway_options = {
        'topleft->botleft',
        'topleft->topright',
        'topright->topleft',
        'topright->botright',
        'botleft->topleft',
        'botleft->botright',
        'botright->botleft',
        'botright->topright'
    }
    
    def __init__(self, name: str, state_space_dim: tuple):
        assert name in self.hallway_options
        self.option_name = name
        self.state_space_dim = state_space_dim
        
        assert self.state_space_dim == (3, 19, 19)
        
        # Instantiate a sub-room in which the option operates
        self.env = gym.make('MiniGrid-Empty-10x10-v0')
        self.env.actions = Actions
        self.env.action_space = gym.spaces.Discrete(len(self.env.actions))
        self.env.max_steps = 100000
        self.reset()
        
        cached_options_dir = f'{ROOT_DIR}/cache/given_options_4rooms'
        if os.path.exists(f'{cached_options_dir}/{name}.pkl'):
            with open(f'{cached_options_dir}/{name}.pkl', 'rb') as f:
                initiation_set, termination_set, policy = pickle.load(f)
        else:
            initiation_set, termination_set, policy = self._create_option()
            with open(f'{cached_options_dir}/{name}.pkl', 'wb') as f:
                pickle.dump((initiation_set, termination_set, policy), f)
        
        super().__init__(initiation_set=initiation_set,
                         termination_function=lambda s: termination_set[s],
                         policy=lambda s: int(policy[s]),
                         name=name)
    
    def __str__(self):
        return self.option_name
    
    def reset(self):
        self.env.reset()
        self._make_hallways()
        self.env.grid.set(8, 8, None)  # Replace default goal by a generic tile
    
    def _make_hallways(self):
        """ Creates a sub-room with two hallways and defines one of them
        to be the goal for the option within the room """
        
        hallways = {
            'topleft->topright' : ((9, 4), (3, 9)),
            'topleft->botleft'  : ((3, 9), (9, 4)),
            'topright->topleft' : ((0, 4), (7, 9)),
            'topright->botright': ((7, 9), (0, 4)),
            'botleft->topleft'  : ((3, 0), (9, 5)),
            'botleft->botright' : ((9, 5), (3, 0)),
            'botright->topright': ((7, 0), (0, 5)),
            'botright->botleft' : ((0, 5), (7, 0)),
        }
        goal, other_hall = hallways[self.option_name]
        self.env.grid.set(*goal, Goal())
        self.env.grid.set(*other_hall, Wall())
    
    def _greedify(self, action, epsilon: float = 0.1):
        if random.random() < epsilon:
            action = random.randint(0, len(self.primitive_options) - 1)
        return action
    
    def _q_learning(self, q_values: np.array = None, epsilon: float = 0.1,
                    alpha: float = 0.1, gamma: float = 0.9):
        """ learns the optimal policy to get to the hallway from anywhere within a room """
        
        # State is number of cells in the grid plus the direction of agent
        # Actions are primitive {left, right, forward}
        if q_values is None:
            q_values = np.zeros((3, 4, 10, 10))
        
        state = (self.env.agent_dir, *self.env.agent_pos,)
        done = False
        
        while not done:
            # self.env.render()
            # time.sleep(0.0005)
            
            a = randargmax(q_values[:, state[0], state[1], state[2]])
            a = self._greedify(a, epsilon)
            obs, reward, done, info = self.env.step(a)
            
            # Note: we could infer the state of the agent from obs, but get it directly instead
            state_next = (self.env.agent_dir, *self.env.agent_pos)
            a_next = randargmax(
                q_values[:, state_next[0], state_next[1], state_next[2]])
            
            q_index = a, state[0], state[1], state[2]
            q_index_next = a_next, state_next[0], state_next[1], state_next[2]
            q_values[q_index] += alpha * (
                reward + gamma * (q_values[q_index_next]) - q_values[q_index])
            
            state = state_next
        
        return q_values
    
    def _mc(self,
            R: np.array = None,
            P: np.array = None,
            N: np.array = None,
            gamma: float = 0.9,
            render: bool = False):
        """ learn model of the option """
        
        # State is number of cells in the grid plus the direction of agent
        # Options consist of primitive {left, right, forward} and multi-step hallway options
        
        if R is None:
            state_space = (4, 10, 10)
            state_action_space = (3, *state_space)
            
            R = np.zeros(state_action_space)
            N = np.zeros(state_action_space)
            P = np.zeros((3, *state_space, *state_space))
        
        self.env.reset()
        state = (self.env.agent_dir, *reversed(self.env.agent_pos))
        done = False
        
        while not done:
            
            a = self.take_option(state)
            obs, reward, done, info = self.env.step(a)
            
            if render:
                print(
                    f"State: {state}, Option: {self.current_option}, Action: {a}")
                self.env.render()
                time.sleep(0.05)
            
            # Note: we could infer the state of the agent from obs, but get it directly instead
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))
            
            # Check for termination condition and update the model
            if self.current_option.termination_set[state_next] == 1:
                option_state = (
                    self.options.index(self.current_option),
                    *self.starting_state)
                # Update visitation counter
                N[option_state] += 1
                alpha = (1 / N[option_state])
                
                # Update reward matrix
                R[option_state] += alpha * (
                    self.cumulative_reward - R[option_state])
                
                # Update probability transition matrix
                P[(*option_state, *state_next)] += alpha * (gamma ** self.k)
                P[option_state] -= alpha * P[option_state]
                
                self.reset_option()
            
            state = state_next
        
        return N, R, P
    
    def _create_option(self):
        """ Defines initiation set, termination set and policy
        for hallway option based on the room the option belongs to """
        
        directions, width, height = dim = self.state_space_dim
        
        initiation_set = np.zeros(dim)
        termination_set = np.ones(dim)
        π = np.zeros(dim) - 1
        
        start_room, goal_room = self.option_name.split('->')
        room_dim = 9
        
        # Compute action-values to determine the policy
        q_values = None
        for _ in tqdm(range(10000)):
            self.reset()
            q_values = self._q_learning(q_values)
        
        policy = np.argmax(q_values, axis=0)
        policy = np.transpose(policy, axes=(0, 2, 1))
        
        # Add hallways to the initialization set
        init_set = {
            'topleft->topright' : (9, 3),
            'topleft->botleft'  : (4, 9),
            'topright->topleft' : (9, 16),
            'topright->botright': (4, 9),
            'botleft->topleft'  : (14, 9),
            'botleft->botright' : (9, 3),
            'botright->topright': (14, 9),
            'botright->botleft' : (9, 16),
        }
        current_room, goal_room = self.option_name.split('->')
        h1, v1 = current_room[:3], current_room[3:]
        h2, v2 = goal_room[:3], goal_room[3:]
        
        if h1 == h2:
            initiation_set[(1, *init_set[self.option_name])] = 1
            initiation_set[(3, *init_set[self.option_name])] = 1
        else:
            initiation_set[(0, *init_set[self.option_name])] = 1
            initiation_set[(2, *init_set[self.option_name])] = 1
        
        if start_room == 'topleft':
            initiation_set[:, 1:room_dim, 1:room_dim] = 1
            termination_set[:, 1:room_dim, 1:room_dim] = 0
            π[:, 0:10, 0:10] = policy
        
        elif start_room == 'topright':
            initiation_set[:, 1:room_dim, room_dim + 1:height - 1] = 1
            termination_set[:, 1:room_dim, room_dim + 1:height - 1] = 0
            π[:, 0:10, 9:19] = policy
        
        elif start_room == 'botleft':
            initiation_set[:, room_dim + 1:height - 1, 1:room_dim] = 1
            termination_set[:, room_dim + 1:height - 1, 1:room_dim] = 0
            π[:, 9:19, 0:10] = policy
        
        elif start_room == 'botright':
            initiation_set[:, room_dim + 1:height - 1,
            room_dim + 1:height - 1] = 1
            termination_set[:, room_dim + 1:height - 1,
            room_dim + 1:height - 1] = 0
            π[:, 9:19, 9:19] = policy
        
        # Add policy for hallway states
        state = list(init_set[self.option_name])
        # if current_room == 'topleft':
        #     pass
        # if current_room == 'topright':
        #     state[1] -= 9
        # elif current_room == 'botleft':
        #     state[0] -= 9
        # elif current_room == 'botright':
        #     state[0] -= 9
        #     state[1] -= 9
        
        if self.option_name == 'botleft->botright':
            π[(0, *state)] = 1
            π[(1, *state)] = 2
            π[(2, *state)] = 0
            π[(3, *state)] = 0
        elif self.option_name == 'botleft->topleft':
            π[(0, *state)] = 0
            π[(1, *state)] = 1
            π[(2, *state)] = 2
            π[(3, *state)] = 0
        elif self.option_name == 'topleft->topright':
            π[(0, *state)] = 0
            π[(1, *state)] = 0
            π[(2, *state)] = 1
            π[(3, *state)] = 2
        elif self.option_name == 'topleft->botleft':
            π[(0, *state)] = 0
            π[(1, *state)] = 1
            π[(2, *state)] = 2
            π[(3, *state)] = 0
        elif self.option_name == 'botright->botleft':
            π[(0, *state)] = 1
            π[(1, *state)] = 2
            π[(2, *state)] = 0
            π[(3, *state)] = 0
        elif self.option_name == 'botright->topright':
            π[(0, *state)] = 2
            π[(1, *state)] = 0
            π[(2, *state)] = 0
            π[(3, *state)] = 1
        elif self.option_name == 'topright->topleft':
            π[(0, *state)] = 0
            π[(1, *state)] = 0
            π[(2, *state)] = 1
            π[(3, *state)] = 2
        elif self.option_name == 'topright->botright':
            π[(0, *state)] = 2
            π[(1, *state)] = 0
            π[(2, *state)] = 0
            π[(3, *state)] = 1
        
        return initiation_set, termination_set, π
