import random
import time
from typing import List, NamedTuple

import numpy as np
import torch
from gym_minigrid.minigrid import MiniGridEnv

from hrl.feature_generators.feature_generators import FeatureGenerator
from hrl.frameworks.gvf.gvf import ControlDemon, TabularQ
from hrl.frameworks.options.option import Option
from hrl.frameworks.options.policies import PolicyOverOptions
from hrl.models.torch.network_bodies import FCBody
from hrl.project_logger import ProjectLogger
from hrl.utils import LearningRate


class Transition(NamedTuple):
    s0: np.ndarray  # current state
    o: Option  # could be a primitive option a well as a learned one
    r: np.ndarray  # reward (does not have to be scalar)
    s1: np.ndarray  # next state
    done: bool  # episode termination indicator


class TorchTransition(NamedTuple):
    s0: torch.Tensor  # current state
    o: Option  # could be a primitive option a well as a learned one
    r: torch.Tensor  # reward (does not have to be scalar)
    s1: torch.Tensor  # next state
    done: bool  # episode termination indicator
    φ0: torch.Tensor  # feature vector of the current state
    φ1: torch.Tensor  # feature vector of the next state
    r: torch.Tensor  # reward (does not have to be scalar)
    Q: torch.Tensor  # Q-values for the current state
    target_Q: torch.Tensor  # Q-values of the next state under target network
    π: torch.Tensor  # action distribution of option's `o` policy in state `s`
    β: torch.Tensor  # probability of terminating in the next state under `o`


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
        
        state = self.env.reset()
        done = False
        executing_option = None
        
        while not done:
            
            if executing_option is None:
                executing_option = self.choose_option(state)
            
            a = executing_option.policy(state)
            s_next, reward, done, info = self.env.step(a)
            
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


class IntraOptionQLearning(ControlDemon):
    """
        We turn now to the intra-option learning of option values and thus of optimal policies
        over options. If the options are semi-Markov, then again the SMDP methods described in
        Section 5.2 are probably the only feasible methods; a semi-Markov option must be completed
        before it can be evaluated in any way. But if the options are Markov and we are willing to
        look inside them, then we can consider intra-option methods. Just as in the case of model
        learning, intra-option methods for value learning are potentially more efficient than SMDP
        methods because they extract more training examples from the same experience.
    """
    
    def __init__(self,
                 env: MiniGridEnv,
                 options: List[Option],
                 target_policy: PolicyOverOptions,
                 feature_generator: FeatureGenerator,
                 weights,  # TODO: type?
                 lr: LearningRate = LearningRate(1, 1, 0),
                 gamma: float = 0.9,
                 loglevel: int = 20,
                 ):
        self.env = env
        # TODO: record all the rewards at the SMDP level?
        self.cumulant = lambda: 0
        
        super().__init__(target_policy=target_policy,
                         termination=lambda: 1,
                         eligibility=lambda: 1,
                         cumulant=self.cumulant,
                         feature=feature_generator,
                         behavioural_policy=target_policy,
                         weights=weights,
                         id=repr(self))
        
        self.options = self.Ω = options
        self.option_idx_dict = {str(o): i for i, o in enumerate(self.options)}
        
        self.lr = lr
        self.gamma = gamma
        
        self.logger = ProjectLogger(level=loglevel, printing=False)
    
    def advantage(self, state):
        """ Used as an unbiased estimator with reduced variance. """
        Q = self.predict(state)
        return Q - np.dot(self.π.pmf(state, Q), Q)
    
    def utility(self, state, option: Option) -> float:
        """ Utility of persisting with the same option vs picking another. """
        ω = self.option_idx_dict[str(option)]
        β = option.β.pmf(state)
        Q = self.predict(state)
        continuation_value = (1 - β) * Q[ω]
        termination_value = β * np.dot(self.π.pmf(state, Q), Q)
        return continuation_value + termination_value
    
    def loss(self,
             s0: np.ndarray,
             o: Option,
             r: float,
             s1: np.ndarray,
             done: bool):
        """ Calculates an Intra-Option Q-learning loss """
        # TODO: rewrite in terms of experience instead
        γ = self.gamma
        ω = self.option_idx_dict[str(o)]
        
        δ = r - self.predict(s0)[ω]
        if not done:
            δ += γ * self.utility(s1, o)
        return δ
    
    def update(self, experience: Transition):
        raise NotImplementedError
    
    def learn_option_values(self, render: bool = False):
        
        env = self.env.unwrapped
        state = self.env.reset()
        executing_option = self.target_policy(state)
        done = False
        while not done:
            
            # Step through environment
            a = executing_option.policy(state)
            s_next, reward, done, info = self.env.step(a)
            
            action_name = list(env.actions)[a].name
            
            # TODO: structure experience in (s, a, r, s') tuples
            self.logger.debug(f"State: {state}, "
                              f"Option: {executing_option}, "
                              f"Action: {action_name}, "
                              f"Next State: {s_next}")
            
            if render:
                self.env.render()
                time.sleep(0.05)
            
            # Update option-values for every option consistent with `a`
            for option in self.options:
                if option.policy(state) == a:
                    self.update(state, reward, s_next, option, done)
            
            # Terminate the option
            if executing_option.termination(s_next) or done:
                executing_option = self.target_policy(s_next)
            
            # Reset the state
            state = s_next
        
        return self.env.step_count


class LinearIntraOptionQLearning(TabularQ, IntraOptionQLearning):
    
    # TODO: figure out the tabular case vs linear with aggregation
    
    def __init__(self, n_states, n_actions, *args, **kwargs):
        super().__init__(n_states, n_actions, *args, **kwargs)
        self.target_w = self.w
    
    def update(self, experience: Transition):
        """ Semi-gradient TD(0) update """
        α = self.lr.rate
        δ = self.loss(*experience)
        ω = self.option_idx_dict[str(experience.o)]
        if δ:
            self.w += α * δ * self.feature(s=experience.s0, o=ω)


class IntraOptionDeepQLearning(IntraOptionQLearning):
    
    def __init__(self,
                 critic: FCBody,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic = critic
        self.advantage_estimator = 'io'
    
    def advantage(self, state, Q: torch.Tensor = None):
        """ Used as an unbiased estimator with reduced variance. """
        return Q - Q.max()
        # return Q - torch.matmul(self.π.pmf(state, action_values=Q), Q)
    
    def estimate_advantage(self, exp: TorchTransition):
        ω = self.option_idx_dict[str(exp.o)]
        if self.advantage_estimator == 'io':
            # Intra-Option Advantage Estimator
            Q_prime = self.predict(exp.φ1)[:, ω]
            advantage = exp.r + self.gamma * (1 - exp.done) * Q_prime - exp.Q[:, ω]
        else:
            raise ValueError(f'Unknown estimator {self.advantage_estimator}')
        return advantage
    
    def loss(self, exp: TorchTransition):
        ω = self.option_idx_dict[str(exp.o)]
        γ = self.gamma
        U = (1 - exp.β) * exp.target_Q[:, ω] + exp.β * torch.max(exp.target_Q)
        loss = (exp.r + γ * (1 - exp.done) * U - exp.Q[:, ω]).pow(2).mul(0.5)
        return loss
    
    def update(self, experience: Transition):
        pass  # auto-grad
    
    def predict(self, obs, *args, **kwargs) -> np.array:
        return self.critic(self.feature(obs))


class IntraOptionActionLearning(IntraOptionQLearning):
    
    def __init__(self, Q_omega, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q_omega = Q_omega
        self.weights = np.zeros((len(self.options), self.env.action_space.n,
                                 *self.env.observation_space.shape))
    
    def predict(self, state, option=slice(None), action=slice(None)):
        return np.dot(self.weights, state)[option, action]
    
    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               s_next: np.ndarray,
               option: Option,
               done: bool):
        """ Preforms an Intra-Option Q-learning update """
        γ = self.gamma
        α = self.lr.rate
        ω = self.option_idx_dict[str(option)]
        
        δ = reward - self.weights[ω, action]
        if not done:
            Q = self.Q_omega.predict(s_next)
            β = option.termination.pmf(s_next)
            δ += γ * ((1 - β) * Q[ω] + β * Q.max())
        self.weights[ω, action] += α * δ
