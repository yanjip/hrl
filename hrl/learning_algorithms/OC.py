import numpy as np
import random
import time
from gym_minigrid.minigrid import MiniGridEnv
from hrl.utils import randargmax
from typing import List

import numpy as np
from scipy.special import expit, logsumexp


class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros(nfeatures)
        # self.weights = np.random.rand(nfeatures)
    
    def pmf(self, phi):
        """ Returns probability of continuing in the current state """
        return expit(np.sum(self.weights[phi]))
    
    def sample(self, phi):
        """ Returns a boolean to indicate whether the option should terminate in a given state """
        return self.rng.uniform() < self.pmf(phi)
    
    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate * (1. - terminate)


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1e-2):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        # self.weights = np.random.rand(nfeatures, nactions)
        self.temp = temp
    
    def value(self, phi, action=None):
        """ Returns a vector of action values for the current state """
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)
    
    def pmf(self, phi):
        """ Returns a vector of probabilities over actions in the current state """
        v = self.value(phi) / self.temp
        return np.exp(v - logsumexp(v))
    
    def sample(self, phi):
        """ Samples an action wrt the current weights """
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))


class OneStepTermination:
    def sample(self, phi):
        return 1
    
    def pmf(self, phi):
        return 1.


class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]
    
    def sample(self, phi):
        return self.action
    
    def pmf(self, phi):
        return self.probs


class FixedOption:
    def __init__(self, action, n_actions):
        self.policy = FixedActionPolicies(action, n_actions)
        self.beta = OneStepTermination()


class LearnedOption:
    
    def __init__(self, n_features, n_actions, rng=np.random.RandomState(1337)):
        self.n_states = n_features
        self.n_actions = n_actions
        
        # Each option has its own policy and termination functions learned via policy gradients
        self.policy = SoftmaxPolicy(rng, n_features, n_actions)
        self.beta = SigmoidTermination(rng, n_features)


class OptionCritic:
    
    def __init__(self,
                 env: MiniGridEnv,
                 options: List[LearnedOption],
                 gamma: float = 0.99,
                 alpha_critic: float = 0.5,
                 alpha_theta: float = 0.25,
                 alpha_upsilon: float = 0.25):
        
        # Environment setup
        self.env = env
        
        # Discount factor
        self.gamma = gamma
        
        # Learning rates
        self.alpha_critic = alpha_critic
        self.alpha_theta = alpha_theta
        self.alpha_upsilon = alpha_upsilon
        
        # Learned options
        self.options = options
        self.last_action = None
        self.last_option = None
        
        n_options = len(options)
        n_states = self.options[0].n_states
        n_actions = self.options[0].n_actions
        
        # Executing action-value function, i.e action values in the context of (state, option) pairs
        self.Q_U = np.zeros((n_states, n_options, n_actions))
        
        # Option-values
        self.Q = np.zeros((n_states, n_options))
        
        # State-values in case e-greedy options learning policy is used
        self.V = np.zeros(n_states)
    
    def choose_action(self, option, state):
        """ Selects action according to the policy of the current option """
        return option.policy.sample(state)
    
    def choose_option(self, state, epsilon=0.01):
        """ Picks the optimal option in an epsilon-greedy way """
        if random.random() > epsilon:
            option = self.options[randargmax(self.Q[state])]
        else:
            option = random.choice(self.options)
        return option
    
    def critic_update(self, state, reward, state_next):
        """ Learns policy over options via Intra-Option Q-learning """
        o = self.options.index(self.last_option)
        a = self.last_action
        phi = np.array([state_next, ])
        
        # Update option-value function Q_omega via Intra-Option Q-learning
        beta = self.last_option.beta.pmf(phi)
        continuation_value = (1 - beta) * self.Q[state_next, o]
        termination_value = beta * np.max(self.Q[state_next])
        target = reward + self.gamma * (continuation_value + termination_value)
        self.Q[state, o] += self.alpha_critic * (target - self.Q[state, o])
        # Note: alternatively use
        #  phi = np.array([state, ])
        #  probs = self.last_option.policy.pmf(phi)
        #  self.Q[state, o] = probs.dot(self.Q_U[state, o])
        
        # Update Q_u via Intra-Option-Action Q-Learning
        self.Q_U[state, o, a] += self.alpha_critic * (
                target - self.Q_U[state, o, a])
        
        # Update state-value function V
        self.V[state] = np.max(self.Q[state, o])
    
    def actor_update(self, option, action, state_next, baseline=True):
        """ Learns an option-specific policy and termination function via policy gradients """
        o = self.options.index(option)
        a = action
        phi = np.array([state_next, ])
        
        # Intra option policy update
        actions_pmf = option.policy.pmf(phi)
        critic = self.Q_U[(state_next, o, a)]
        if baseline:
            critic -= self.Q[(state_next, o)]
        option.policy.weights[phi] -= self.alpha_theta * critic * actions_pmf
        option.policy.weights[phi, a] += self.alpha_theta * critic
        
        # Termination function update
        magnitude = option.beta.grad(phi)
        advantage = self.Q[state_next, o] - self.V[state_next]
        option.beta.weights[phi] -= self.alpha_upsilon * advantage * magnitude
    
    def get_state(self):
        return (self.env.agent_dir, *reversed(self.env.agent_pos))
    
    def run_episode(self, render: bool = False):
        
        # Initialize
        self.env.reset()
        state = self.state_index(self.get_state())
        option = self.last_option = self.choose_option(state)
        self.last_action = self.choose_action(option, np.array([state, ]))
        done = False
        
        # Trackers
        cumreward = 0.
        duration = 1
        option_switches = 0
        avgduration = 0.
        steps = 0
        
        while not done:
            steps += 1
            if render:
                self.env.render()
                time.sleep(0.05)
            
            # Take action, observe next state and reward
            obs, reward, done, info = self.env.step(self.last_action)
            state_next = self.state_index(self.get_state())
            phi = np.array([state_next, ])
            
            # Choose another option in case the current one terminates
            if option.beta.sample(phi):
                # print(f'Option: {self.options.index(option)}, PMF: {option.beta.pmf(phi)}')
                self.last_option = option
                option = self.choose_option(state_next)
                option_switches += 1
                avgduration += (1. / option_switches) * (duration - avgduration)
                duration = 1
            
            # Choose next action according to the intra-option policy of the current option
            action = self.choose_action(option, phi)
            
            # Evaluate and improve options
            
            self.critic_update(state, reward, state_next)
            
            if isinstance(option, LearnedOption):
                self.actor_update(option, action, state_next)
            
            # Update trackers
            self.last_action = action
            
            state = state_next
            cumreward += reward
            duration += 1

        np.set_printoptions(precision=3, suppress=True, linewidth=150)
        print(np.mean(self.Q.reshape((4, 19, 19, len(self.options))), axis=(0,3 )))
        print(f'steps {steps} cumreward {round(cumreward, 2)} '
              f'avg. duration {round(avgduration, 2)} switches {option_switches}')
    
    def state_index(self, state):
        # matrix[ i ][ j ][ k ] = array[ i*(N*M) + j*M + k ]
        i = state[0]
        i *= 19
        i += state[1]
        i *= 19
        i += state[2]
        return i


class OptionCriticNetwork:
    pass
