import random
import time
from typing import List

import numpy as np
from gym_minigrid.minigrid import MiniGridEnv

from hrl.frameworks.options.option import Option, MarkovOption
from hrl.project_logger import ProjectLogger
from hrl.utils import randargmax

Option = List[Option]


class IntraOptionModelLearning:
    
    def __init__(self,
                 env: MiniGridEnv,
                 options: Option,
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
                    N: np.ndarray = None,
                    R: np.ndarray = None,
                    P: np.ndarray = None,
                    γ: float = 0.9,
                    α: float = None,
                    render: bool = False):
    
        env = self.env.unwrapped
        n_options = len(self.options)
        state_space = (4, env.width, env.height)
        dim = (n_options, *state_space)
        
        if R is None:
            R = np.zeros(dim)
            N = np.zeros(dim)
            P = np.zeros((n_options, *state_space, *state_space))
        
        self.env.reset()
        state = (env.agent_dir, *reversed(env.agent_pos))
        done = False
        executing_option = None
        
        while not done:
    
            if executing_option is None:
                executing_option = self.choose_option(state)
    
            a = executing_option.policy(state)
            obs, reward, done, info = self.env.step(a)
            s_next = (env.agent_dir, *reversed(env.agent_pos))
            
            if render:
                action_name = list(env.actions)[a].name
                self.logger.debug(f"State: {state}, "
                                  f"Option: {executing_option}, "
                                  f"Action: {action_name}, "
                                  f"Next State: {s_next}")
                self.env.render()
                time.sleep(0.05)
            
            # Update model for every option consistent with last action taken
            for option in self.options:
                # if option.name != executing_option.name:
                #     continue
                
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
            yield N, R, P
        
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

    def __init__(self,
                 env: MiniGridEnv,
                 options: Option,
                 loglevel: int = 20):
        self.env = env
        self.options = options
        self.option_names_dict = {o.name: o for o in self.options}
        self.option_idx_dict = {name: i for i, name in
                                enumerate(self.option_names_dict)}
        self.logger = ProjectLogger(level=loglevel, printing=True)

    def _epsilon_greedy(self,
                        Q: np.ndarray,
                        state: tuple,
                        ε: float = 0.1
                        ) -> Option:
        """ Picks the optimal option with probability 1 - ε """
    
        # Filter options based on their initiation set
        options = list(filter(lambda o: o[1].initiation_set[state] == 1,
                              enumerate(self.options)))
        option_indices = [i for i, o in options]
        self.logger.debug(f'Available options: {[str(o) for i, o in options]}')
    
        # Get an option with max Q
        all_options = Q[(slice(None), *state)]
        max_q = all_options[option_indices].max()
        
        # Pick at random from the best available options
        valid = set(option_indices) & set(np.where(all_options == max_q)[0])
        optimal_option = self.options[random.choice(list(valid))]
        self.logger.debug(f'Best option: {optimal_option}')
    
        # Add randomness to the choice
        if random.random() > ε:
            return optimal_option
        else:
            return random.choice([self.options[i] for i, o in options])

    def choose_option(self,
                      Q: np.ndarray,
                      state: tuple,
                      policy: str = 'epsilon_greedy',
                      ε: float = 0.1) -> MarkovOption:
        if policy == 'epsilon_greedy':
            option = self._epsilon_greedy(Q, state, ε)
        else:
            raise ValueError(f'Policy {policy} is not implemented')
    
        return MarkovOption(
            starting_state=state,
            initiation_set=option.initiation_set,
            termination_set=option._termination_set,
            policy=option._policy,
            name=str(option)
        )

    def q_learning(self,
                   n_episodes: int,
                   γ: float = 0.9,
                   Q: np.ndarray = None,
                   ε: float = 0.1,
                   α: float = 1 / 4,
                   render: bool = False):
    
        env = self.env.unwrapped
        n_options = len(self.options)
        state_space_dim = (4, env.width, env.height)
        dim = (n_options, *state_space_dim)
    
        if Q is None:
            Q = np.zeros(dim)
    
        for episode in range(n_episodes):
        
            self.env.reset()
            state = (env.agent_dir, *reversed(env.agent_pos))
            executing_option = self.choose_option(Q, state, ε=ε)
            done = False
            self.logger.debug(f'{executing_option}, {state}')
            while not done:
            
                # Step through environment
                a = executing_option.policy(state)
                try:
                    obs, reward, done, info = self.env.step(a)
                except Exception as e:
                    print(e)
                    print(a)
                    assert 0
            
                # TODO: infer the state of the agent from obs
                s_next = (env.agent_dir, *reversed(env.agent_pos))
                o_next = randargmax(Q[(slice(None), *s_next)])
            
                if True:
                    action_name = list(env.actions)[a].name
                    self.logger.info(f"State: {state}, "
                                     f"Option: {executing_option}, "
                                     f"Action: {action_name}, "
                                     f"Next State: {s_next}, "
                                     f"Next Option: {self.options[o_next]}")
                    self.env.render()
                    time.sleep(0.05)
            
                # Update action-values for every option consistent with `a`
                for option in self.options:
                    if option.policy(state) != a:
                        continue
                
                    option_index = self.option_idx_dict[option.name]
                    option_state = (option_index, *state)
                
                    β = option.termination_function(s_next)
                    continuation_value = (1 - β) * Q[(option_index, *s_next)]
                    termination_value = β * Q[(o_next, *s_next)]
                    U = continuation_value + termination_value
                
                    target = reward + γ * U
                    Q[option_state] += α * (target - Q[option_state])
            
                # Terminate the option
                if executing_option.termination_function(s_next) == 1 or done:
                    executing_option = self.choose_option(Q, state, ε=ε)
            
                # Reset the state
                state = s_next
        
            yield Q, self.env.step_count
    
        return Q, self.env.step_count
