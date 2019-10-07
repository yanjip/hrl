import argparse
import numpy as np
from hrl.envs.four_rooms import FourRooms
import pickle
import redis
import random
from functools import partial

from scipy.special import expit, logsumexp
from hrl.utils import randargmax

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates
    
    def __call__(self, state):
        return np.array([state, ])
    
    def __len__(self):
        return self.nstates


class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))
    
    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)
    
    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return randargmax(self.value(phi))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp
    
    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)
    
    def pmf(self, phi):
        v = self.value(phi) / self.temp
        return np.exp(v - logsumexp(v))
    
    def sample(self, phi):
        p = self.pmf(phi)
        return int(self.rng.choice(self.weights.shape[1], p=p))


class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
    
    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))
    
    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))
    
    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate * (1. - terminate), phi


class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
    
    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)
    
    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)
    
    def advantage(self, phi, option=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if option is None:
            return advantages
        return advantages[option]
    
    def update(self, phi, option, reward, done):
        
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount * (
                (1. - termination) * current_values[self.last_option] + termination * np.max(current_values))
        
        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr * tderror
        
        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi
        
        return update_target


class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega
    
    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)
    
    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action
    
    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount * (
                (1. - termination) * current_values[self.last_option] + termination * np.max(current_values))
        
        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr * tderror
        
        self.last_phi = phi
        self.last_option = option
        self.last_action = action


class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr
    
    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -=\
            self.lr * magnitude * (self.critic.advantage(phi, option))


class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies
    
    def update(self, phi, option, action, critic):
        actions_pmf = self.option_policies[option].pmf(phi)
        self.option_policies[option].weights[phi, :] -= self.lr * critic * actions_pmf
        self.option_policies[option].weights[phi, action] += self.lr * critic


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


def stochastic_step(env, action, prob=0.9):
    env.step_count += 1
    
    reward = 0
    done = False
    
    # Get the position in front of the agent
    fwd_pos = env.front_pos
    
    # Get the contents of the cell in front of the agent
    fwd_cell = env.grid.get(*fwd_pos)
    
    def rotate_left():
        env.agent_dir -= 1
        if env.agent_dir < 0:
            env.agent_dir += 4
        return reward, done
    
    def rotate_right():
        env.agent_dir = (env.agent_dir + 1) % 4
        return reward, done
    
    def forward(reward=0, done=False):
        if fwd_cell is None or fwd_cell.can_overlap():
            env.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            done = True
            reward = env._reward()
        if fwd_cell is not None and fwd_cell.type == 'lava':
            done = True
        
        return reward, done
    
    # Rotate left
    if action == env.actions.left:
        if random.random() > prob:
            reward, done = random.choice((rotate_right, forward))()
        else:
            reward, done = rotate_left()
    
    # Rotate right
    elif action == env.actions.right:
        if random.random() > prob:
            reward, done = random.choice((rotate_left, forward))()
        else:
            reward, done = rotate_right()
    
    # Move forward
    elif action == env.actions.forward:
        if random.random() > prob:
            reward, done = random.choice((rotate_left, rotate_right))()
        else:
            reward, done = forward()
    
    # Done action (not used by default)
    elif action == env.actions.done:
        pass
    
    else:
        assert False, "unknown action"
    
    if env.step_count >= env.max_steps:
        done = True
    
    obs = env.gen_obs()
    
    return obs, reward, done, {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=1e-2)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-2)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=250)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=100000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true',
                        default=False)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')
    
    args = parser.parse_args()
    
    rng = np.random.RandomState(1234)
    
    # Environment setup
    env = FourRooms(agent_pos=(1, 1), goal_pos=(15, 15))
    env.max_steps = 1000000
    env.step = partial(stochastic_step, env)

    db = redis.StrictRedis(port=6379)
    
    
    def get_state():
        return (env.agent_dir, *reversed(env.agent_pos))
    
    
    def state_index(state):
        # matrix[ i ][ j ][ k ] = array[ i*(N*M) + j*M + k ]
        i = state[0]
        i *= 19
        i += state[1]
        i *= 19
        i += state[2]
        return i
    
    
    fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
    fname = 'optioncritic-fourrooms-' + fname + '.npy'
    
    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
    
    history = np.zeros((args.nruns, args.nepisodes, 2))
    
    for run in range(args.nruns):
        
        nfeatures = 4 * 19 * 19
        nactions = 3
        # features = Tabular(env.observation_space.n)
        features = Tabular(nfeatures)
        # nfeatures, nactions = len(features), env.action_space.n
        nfeatures, nactions = len(features), nactions
        
        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for _ in range(args.noptions)]
        if args.primitive:
            option_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])
        
        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(nactions)])
        
        # E-greedy policy over options
        policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        # policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)
        
        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)
        
        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_critic, option_terminations, action_weights,
                                                   critic)
        
        # Improvement of the termination functions based on gradients
        termination_improvement = TerminationGradient(option_terminations, critic, args.lr_term)
        
        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)
        
        critic_decay = critic.lr / 10000
        intra_decay = intraoption_improvement.lr / 5000
        term_decay = termination_improvement.lr / 5000
        for episode in range(args.nepisodes):
            
            # decay learning rates
            critic.lr = max(0.01, critic.lr * (1. / (1. + critic_decay * episode)))
            action_critic.lr = max(0.01, action_critic.lr * (1. / (1. + critic_decay * episode)))
            intraoption_improvement.lr = max(0.001, intraoption_improvement.lr * (1. / (1. + intra_decay * episode)))
            termination_improvement.lr = max(0.001, termination_improvement.lr * (1. / (1. + term_decay * episode)))
            print(f'Q LR: {critic.lr}, PG LR {intraoption_improvement.lr}')
            # if episode == 1000:
            #     env.goal = rng.choice(possible_next_goals)
            #     print('************* New goal : ', env.goal)
            
            env.reset()
            phi = features(state_index(get_state()))
            # phi = features(env.reset())
            option = policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)
            action_critic.start(phi, option, action)
            
            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            for step in range(args.nsteps):
                # if episode >= 100 and episode % 100 == 0:
                #     env.render()
                observation, reward, done, _ = env.step(action)
                observation = state_index(get_state())
                phi = features(observation)
                
                # Termination might occur upon entering the new state
                if option_terminations[option].sample(phi):
                    # print(f'Option: {option}, PMF: {option_terminations[option].pmf(phi)}')
                    option = policy.sample(phi)
                    option_switches += 1
                    avgduration += (1. / option_switches) * (duration - avgduration)
                    duration = 1
                
                action = option_policies[option].sample(phi)
                
                # Critic update
                update_target = critic.update(phi, option, reward, done)
                action_critic.update(phi, option, action, reward, done)
                
                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, option)
                    intraoption_improvement.update(phi, option, action, critic_feedback)
                    
                    # Termination update
                    termination_improvement.update(phi, option)
                
                cumreward += reward
                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = avgduration
            print(f'Run {run} episode {episode} steps {step} cumreward {cumreward} avg. duration {avgduration} '
                  f'switches {option_switches} \n')

        #     dim = (4, 19, 19)
        #     Q = np.zeros((*dim, args.noptions))
        #     Qu = np.zeros((*dim, args.noptions, nactions))
        #     term = np.zeros((args.noptions, *dim))
        #     pi = term.copy()
        #
        #     for i in range(4):
        #         for j in range(19):
        #             for k in range(19):
        #                 state = (i, j, k)
        #                 ix = state_index(state)
        #                 phi = np.array([ix,])
        #                 Q[state] = critic.value(phi)
        #                 Qu[state] = action_critic.weights[phi[0], :, :]
        #                 for o in range(args.noptions):
        #                     term[(o, *state)] = option_terminations[o].pmf(phi)
        #                     pi[(o, *state)] = option_policies[o].sample(phi)
        #
        #     db.set('Options-Q-Values', pickle.dumps(Q))
        #     db.set('Options-Qu-Values', pickle.dumps(Qu))
        #     db.set('Terminations', pickle.dumps(term))
        #     db.set('Policies', pickle.dumps(pi))
        #
        # np.save(fname, history)
        # pickle.dump({'intra_policies': option_policies, 'policy': policy, 'term': option_terminations},
        #           open('oc-options.pl', 'wb'))
        # print(fname)
