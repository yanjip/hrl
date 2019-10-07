import time

from gym_minigrid.minigrid import MiniGridEnv

from hrl.frameworks import GVF
from hrl.frameworks.options import PrimitivePolicy
from hrl.frameworks.options.policies import PolicyOverOptions
from hrl.learning_algorithms.intra_option import IntraOptionQLearning, IntraOptionActionLearning
from hrl.project_logger import ProjectLogger


class OptionCritic:
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
    
    def estimate_advantages(self, state, option: GVF, reward: float, s_next):
        ω = self.actor.option_idx_dict[str(option)]
        if self.advantage_estimator == 'io':
            # Intra-Option Advantage Estimator
            Q = self.critic(s_next, ω)
            advantage = reward + self.γ * Q - self.critic(state, ω)
        elif self.advantage_estimator == 'augmented':
            # Augmented Advantage Estimator
            U = self.critic.utility(state, option)
            advantage = reward + self.γ * U - self.critic(state, ω)
        else:
            raise ValueError(f'Unknown estimator {self.advantage_estimator}')
        return advantage
    
    def learn(self, baseline: bool = False, render: bool = False):
        
        # Trackers
        cumreward = 0.
        duration = 0
        option_switches = 0
        avgduration = 0.
        
        # Initialize s0 and pick an option
        s0 = self.env.reset()
        
        option = self.actor(s0, action_values=self.critic.option_values(s0))
        ω = self.actor.option_idx_dict[str(option)]
        
        # Run until episode termination
        done = False
        while not done:
            if render:
                self.env.render()
                time.sleep(0.05)
            
            # Take a, observe next s0 and reward
            a = option.μ(s0)
            s1, reward, done, info = self.env.step(a)
            
            # Option evaluation step
            self.critic.update(s0, reward, s1, option, done)
            if self.advantage_estimator is None:
                self.action_critic.update(s0, a, reward, s1, option, done)
            
            # Option improvement step
            if not isinstance(option.π, PrimitivePolicy):
                if self.advantage_estimator is None:
                    critique = self.action_critic(s0, ω, a)
                    if baseline:
                        critique -= self.critic(s0, ω)
                else:
                    critique = self.estimate_advantages(s0, option, reward, s1)
                
                option.π.update(s0, a, critique)
                option.γ.update(s1, self.critic.advantage(s1, ω))
            
            # Choose another option in case the current one terminates
            if option.termination(s1):
                option = self.actor(s0, self.critic.option_values(s1))
                ω = self.actor.option_idx_dict[str(option)]
                option_switches += 1
                avgduration += (1. / option_switches) * (duration - avgduration)
                duration = 0
            
            s0 = s1
            cumreward += reward
            duration += 1
        
        self.logger.info(f'steps {self.env.unwrapped.step_count}\n'
                         f'cumreward {round(cumreward, 2)}\n'
                         f'avg. duration {round(avgduration, 2)}\n'
                         f'switches {option_switches}\n'
                         f'critic lr {self.critic.lr.rate}\n'
                         f'')


class OptionCriticNetwork:
    pass
