import gym
import random
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
from enum import IntEnum

from gym_minigrid.minigrid import Floor, Wall, Goal
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.register import register

from hrl.utils import randargmax, ROOT_DIR


class EmptyEnv10x10(EmptyEnv):
    """ A single room to learn the hallway option on """
    
    def __init__(self):
        super().__init__(size=10, agent_start_pos=None)


env_name = 'MiniGrid-Empty-10x10-v1'
if env_name not in gym.envs.registry.env_specs:
    # del gym.envs.registry.env_specs[env_name]
    register(
        id=env_name,
        entry_point=f"hallway_options:EmptyEnv10x10"
    )


class HallwayOption:
    
    def __init__(self, option):
        self.name = option
        
        class Actions(IntEnum):
            # Turn left, turn right, move forward
            left = 0
            right = 1
            forward = 2
            
            # Done completing task
            done = 3
        
        # Instantiate a sub-room in which the option operates
        self.env = gym.make('MiniGrid-Empty-10x10-v1')
        self.env.actions = Actions
        self.env.action_space = gym.spaces.Discrete(len(self.env.actions))
        self.env.max_steps = 100000
        
        self.initiation_set, self.termination_set, self.policy = None, None, None
        
        self.primitive_options = {'left', 'right', 'forward'}
        
        # Create hallways in the room and define one of the hallways to be the goal within the room
        self.goal, self.other_hall = None, None
        self.hallways = {
            'topleft->topright' : ((9, 4), (3, 9)),
            'topleft->botleft'  : ((3, 9), (9, 4)),
            'topright->topleft' : ((0, 4), (7, 9)),
            'topright->botright': ((7, 9), (0, 4)),
            'botleft->topleft'  : ((3, 0), (9, 5)),
            'botleft->botright' : ((9, 5), (3, 0)),
            'botright->topright': ((7, 0), (0, 5)),
            'botright->botleft' : ((0, 5), (7, 0)),
        }
        if option not in self.primitive_options:
            self._reset()
        
        if os.path.exists(f'{ROOT_DIR}/cache/given_options_4rooms/{option}.pkl'):
            with open(f'{ROOT_DIR}/cache/given_options_4rooms/{option}.pkl', 'rb') as f:
                self.initiation_set, self.termination_set, self.policy = pickle.load(f)
        else:
            self._create_option()
            with open(f'{ROOT_DIR}/cache/given_options_4rooms/{option}.pkl', 'wb') as f:
                pickle.dump((self.initiation_set, self.termination_set, self.policy), f)
    
    def choose_action(self, state):
        """ takes action according to the greedy policy learned by Q-learning """
        return int(self.policy[state])
    
    def _reset(self):
        self.env.reset()
        self._make_hallways()
        
        # Replace default goal by a generic tile
        self.env.grid.set(8, 8, Floor())
    
    def _make_hallways(self):
        """ creates a sub-room with two hallways that the option will operate in """
        
        self.goal = self.hallways[self.name][0]
        self.other_hall = self.hallways[self.name][1]
        self.env.grid.set(*self.goal, Goal())
        self.env.grid.set(*self.other_hall, Wall())
    
    def _greedify(self, action, epsilon: float = 0.1):
        if random.random() < epsilon:
            action = random.randint(0, len(self.primitive_options) - 1)
        return action
    
    def _q_learning(self, q_values: np.array = None, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.9):
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
            a_next = randargmax(q_values[:, state_next[0], state_next[1], state_next[2]])
            
            q_index = a, state[0], state[1], state[2]
            q_index_next = a_next, state_next[0], state_next[1], state_next[2]
            q_values[q_index] += alpha * (reward + gamma * (q_values[q_index_next]) - q_values[q_index])
            
            state = state_next
        
        return q_values
    
    def _mc(self, R: np.array = None, P: np.array = None, N: np.array = None, gamma: float = 0.9,
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
                print(f"State: {state}, Option: {self.current_option}, Action: {a}")
                self.env.render()
                time.sleep(0.05)
            
            # Note: we could infer the state of the agent from obs, but get it directly instead
            state_next = (self.env.agent_dir, *reversed(self.env.agent_pos))
            
            # Check for termination condition and update the model
            if self.current_option.termination_set[state_next] == 1:
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
    
    def _create_option(self):
        """ defines initiation set, termination set and policy for the option """
        
        directions, width, height = dim = (4, 19, 19)
        
        if self.name in self.primitive_options:
            # primitive options are defined everywhere and can terminate anywhere
            
            self.initiation_set = np.ones(dim)
            self.termination_set = np.ones(dim)
            
            if self.name == "left":
                self.policy = np.zeros(dim)
            elif self.name == "right":
                self.policy = np.zeros(dim) + 1
            elif self.name == "forward":
                self.policy = np.zeros(dim) + 2
        
        else:
            self.initiation_set = np.zeros(dim)
            self.termination_set = np.ones(dim)
            self.policy = np.zeros(dim) - 1
            
            start_room, goal_room = self.name.split('->')
            room_dim = 9
            
            # Compute action-values to determine the policy
            q_values = None
            for _ in tqdm(range(10000)):
                self._reset()
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
            current_room, goal_room = self.name.split('->')
            h1, v1 = current_room[:3], current_room[3:]
            h2, v2 = goal_room[:3], goal_room[3:]
            
            if h1 == h2:
                self.initiation_set[(1, *init_set[self.name])] = 1
                self.initiation_set[(3, *init_set[self.name])] = 1
            else:
                self.initiation_set[(0, *init_set[self.name])] = 1
                self.initiation_set[(2, *init_set[self.name])] = 1
            
            if start_room == 'topleft':
                self.initiation_set[:, 1:room_dim, 1:room_dim] = 1
                self.termination_set[:, 1:room_dim, 1:room_dim] = 0
                self.policy[:, 0:10, 0:10] = policy
            
            elif start_room == 'topright':
                self.initiation_set[:, 1:room_dim, room_dim + 1:height - 1] = 1
                self.termination_set[:, 1:room_dim, room_dim + 1:height - 1] = 0
                self.policy[:, 0:10, 9:19] = policy
            
            elif start_room == 'botleft':
                self.initiation_set[:, room_dim + 1:height - 1, 1:room_dim] = 1
                self.termination_set[:, room_dim + 1:height - 1, 1:room_dim] = 0
                self.policy[:, 9:19, 0:10] = policy
            
            elif start_room == 'botright':
                self.initiation_set[:, room_dim + 1:height - 1, room_dim + 1:height - 1] = 1
                self.termination_set[:, room_dim + 1:height - 1, room_dim + 1:height - 1] = 0
                self.policy[:, 9:19, 9:19] = policy
            
            # Add policy for hallway states
            state = list(init_set[self.name])
            # if current_room == 'topleft':
            #     pass
            # if current_room == 'topright':
            #     state[1] -= 9
            # elif current_room == 'botleft':
            #     state[0] -= 9
            # elif current_room == 'botright':
            #     state[0] -= 9
            #     state[1] -= 9
            
            if self.name == 'botleft->botright':
                self.policy[(0, *state)] = 1
                self.policy[(1, *state)] = 2
                self.policy[(2, *state)] = 0
                self.policy[(3, *state)] = 0
            elif self.name == 'botleft->topleft':
                self.policy[(0, *state)] = 0
                self.policy[(1, *state)] = 1
                self.policy[(2, *state)] = 2
                self.policy[(3, *state)] = 0
            elif self.name == 'topleft->topright':
                self.policy[(0, *state)] = 0
                self.policy[(1, *state)] = 0
                self.policy[(2, *state)] = 1
                self.policy[(3, *state)] = 2
            elif self.name == 'topleft->botleft':
                self.policy[(0, *state)] = 0
                self.policy[(1, *state)] = 1
                self.policy[(2, *state)] = 2
                self.policy[(3, *state)] = 0
            elif self.name == 'botright->botleft':
                self.policy[(0, *state)] = 1
                self.policy[(1, *state)] = 2
                self.policy[(2, *state)] = 0
                self.policy[(3, *state)] = 0
            elif self.name == 'botright->topright':
                self.policy[(0, *state)] = 2
                self.policy[(1, *state)] = 0
                self.policy[(2, *state)] = 0
                self.policy[(3, *state)] = 1
            elif self.name == 'topright->topleft':
                self.policy[(0, *state)] = 0
                self.policy[(1, *state)] = 0
                self.policy[(2, *state)] = 1
                self.policy[(3, *state)] = 2
            elif self.name == 'topright->botright':
                self.policy[(0, *state)] = 2
                self.policy[(1, *state)] = 0
                self.policy[(2, *state)] = 0
                self.policy[(3, *state)] = 1


if __name__ == '__main__':
    options = {
        'left', 'right', 'forward',
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
