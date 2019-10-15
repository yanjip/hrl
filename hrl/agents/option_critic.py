import time
import numpy as np
import torch
from gym_minigrid.minigrid import MiniGridEnv

from hrl.agents.intra_option import IntraOptionQLearning, IntraOptionActionLearning, Transition, TorchTransition, IntraOptionDeepQLearning
from hrl.frameworks.options import PrimitivePolicy, Option
from hrl.frameworks.options.policies import PolicyOverOptions
from hrl.models.torch.network_bodies import NatureConvBody
from hrl.project_logger import ProjectLogger
from copy import deepcopy
import torch.nn.functional as f

class OptionCriticAgent:
    """  Actor-Critic style agent for learning options in a differentiable way.
    
    Architecture Overview:
    The option policies parameterized by θ, termination functions
    parameterized by υ and policy over options µθ belong to the actor part
    of the system while the critic consists of Q_U (action-value upon entering)
    and A_Ω (advantages of the policy over options).
    Policy over options is learned via Intra-Option Q-Learning, but could
    also be learned using policy gradients at SMDP level.
    The algorithm uses variations of Policy Gradients theorem, to learn option's
    policies and termination functions from a single stream of experience.
    """
    
    # TODO: add deliberation cost
    
    def __init__(self,
                 env: MiniGridEnv,
                 critic: IntraOptionQLearning,
                 actor: PolicyOverOptions,
                 action_critic: IntraOptionActionLearning = None,
                 gamma: float = 0.99,
                 loglevel: int = 20):
        self.env = env
        self.γ = gamma
        self.critic = critic
        self.actor = actor
        self.logger = ProjectLogger(level=loglevel)
        if action_critic is not None:
            self.advantage_estimator = None
            self.action_critic = action_critic
        else:
            self.advantage_estimator = 'io'
            self.logger.info(f'Action-critic was not provided, '
                             f'so the advantages for PG would be estimated')
    
    def estimate_advantages(self, state, option: Option, reward: float, s_next):
        ω = self.actor.option_idx_dict[str(option)]
        if self.advantage_estimator == 'io':
            # Intra-Option Advantage Estimator
            Q = self.critic(s_next)[ω]
            advantage = reward + self.γ * Q - self.critic(state)[ω]
        elif self.advantage_estimator == 'augmented':
            # Augmented Advantage Estimator
            # FIXME: utility should be calculated wrt to the next option!
            U = self.critic.utility(state, option)
            advantage = reward + self.γ * U - self.critic(state)[ω]
        else:
            raise ValueError(f'Unknown estimator {self.advantage_estimator}')
        return advantage
    
    def learn(self, baseline: bool = False, render: bool = False):
        
        # Trackers
        cumulant = 0.
        duration = 0
        option_switches = 0
        avgduration = 0.
        
        # Initialize s0 and pick an option
        s0 = self.env.reset()
        
        option = self.actor(s0, action_values=self.critic(s0))
        ω = self.actor.option_idx_dict[str(option)]
        
        # Run until episode termination
        done = False
        while not done:
            if render:
                self.env.render()
                time.sleep(0.05)
            
            # Take action (a), observe next state (s1) and reward (r)
            a = option.π(s0)
            s1, r, done, _ = self.env.step(a)
            experience = Transition(s0, option, r, s1, done)
            
            # Option evaluation step
            self.critic.update(experience)
            if self.advantage_estimator is None:
                self.action_critic.update(s0, a, r, s1, option, done)
            
            # Option improvement step
            if not isinstance(option.π, PrimitivePolicy):
                if self.advantage_estimator is None:
                    critique = self.action_critic(s0, ω, a)
                    if baseline:
                        critique -= self.critic(s0)[ω]
                else:
                    critique = self.estimate_advantages(s0, option, r, s1)
                
                if critique:
                    option.π.update(s0, a, critique)
                
                termination_advantage = self.critic.advantage(s1)[ω]
                if termination_advantage:
                    option.β.update(s1, termination_advantage)
            
            # Choose another option in case the current one terminates
            if option.termination(s1):
                option = self.actor(s1, action_values=self.critic(s1))
                ω = self.actor.option_idx_dict[str(option)]
                option_switches += 1
                avgduration += (1. / option_switches) * (duration - avgduration)
                duration = 0
            
            s0 = s1
            cumulant += r
            duration += 1
        
        self.logger.info(f'steps {self.env.unwrapped.step_count}\n'
                         f'cumulant {round(cumulant, 2)}\n'
                         f'avg. duration {round(avgduration, 2)}\n'
                         f'switches {option_switches}\n'
                         f'critic lr {self.critic.lr.rate}\n'
                         f'')


class OptionCriticNetwork:
    """
    
    Here the weights for both critic and actor are learned by a single NN
    with multiple heads.
    """
    def __init__(self,
                 env: MiniGridEnv,
                 feature_generator: NatureConvBody,
                 critic: IntraOptionDeepQLearning,
                 actor: PolicyOverOptions,
                 optimizer: torch.optim.Optimizer,
                 gamma: float = 0.99,
                 loglevel: int = 20,
                 rng: np.random.RandomState = np.random.RandomState(1),
                 ):
        self.env = env
        self.γ = gamma
        # shared between both critic and actor
        self.feature_generator = feature_generator
        self.critic = critic
        self.actor = actor
        # self.network = torch.nn.Sequential(self.feature_generator, self.critic.critic)
        self.target_network = deepcopy(self.critic.critic)
        self.optimizer = optimizer
        self.logger = ProjectLogger(level=loglevel)
        self.rng = rng
        # TODO: check if the weights of the `network` change
        # self.target_network.load_state_dict(self.network.state_dict())
        
    def learn(self, config):
        
        # Trackers
        cumulant = 0.
        duration = 0
        option_switches = 0
        avgduration = 0.
        
        # Initialize
        s0 = self.env.reset()
        s0 = f.normalize(s0, dim=(2, 3))
        φ0 = self.feature_generator(s0)
        Q = self.critic(φ0)
        option = self.actor(φ0, action_values=Q)
        ω = self.actor.option_idx_dict[str(option)]

        # Run until episode termination
        done = False
        while not done:
            π = option.π(φ0)
            dist = torch.distributions.Categorical(probs=π)
            action, entropy = dist.sample(), dist.entropy()
            
            s1, r, done, info = self.env.step(int(action))
            s1 = f.normalize(s1, dim=(2, 3))
            φ1 = self.feature_generator(s1)
            with torch.no_grad():
                target_Q = self.target_network(φ1)
                β = option.termination(φ1)
            experience = TorchTransition(s0=s0, o=option, r=r, s1=s1, done=done,
                                         φ0=φ0, φ1=φ1, Q=Q, target_Q=target_Q,
                                         π=π, β=β)
            q_loss = self.critic.loss(experience)
            
            critique = self.critic.estimate_advantage(experience)
            pi_loss = -(torch.log(π)[:, action] * critique.detach()) - config.entropy_weight * entropy

            termination_advantage = self.critic.advantage(φ1, Q)[:, ω]
            beta_loss = β * (termination_advantage.detach() + config.η)
            
            self.optimizer.zero_grad()
            (pi_loss + q_loss + beta_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.feature_generator.parameters(),
                                           config.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.critic.parameters(),
                                           config.gradient_clip)
            torch.nn.utils.clip_grad_norm_(option.π.parameters(),
                                           config.gradient_clip)
            self.optimizer.step()
            
            # Choose another option in case the current one terminates
            Q = self.critic(φ1)
            if β > self.rng.uniform():
                option = self.actor(φ1, action_values=Q)
                ω = self.actor.option_idx_dict[str(option)]
                option_switches += 1
                avgduration += (1. / option_switches) * (duration - avgduration)
                duration = 0
                
            s0 = s1
            φ0 = φ1
            
            if self.env.step_count % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.critic.critic.state_dict())
                
            cumulant += r
            duration += 1
        
        self.logger.info(f'steps {self.env.unwrapped.step_count}\n'
                         f'cumulant {round(cumulant, 2)}\n'
                         f'avg. duration {round(avgduration, 2)}\n'
                         f'switches {option_switches}\n'
                         f'critic lr {self.critic.lr.rate}\n'
                         f'')
