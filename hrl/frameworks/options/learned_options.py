import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp


class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        # self.weights = np.zeros(nfeatures)
        self.weights = np.random.rand(nfeatures)
    
    def pmf(self, phi):
        """ Returns probability of continuing in the current state """
        return expit(np.sum(self.weights[phi]))
    
    def terminate(self, phi):
        """ Returns a boolean to indicate whether the option should terminate in a given state """
        return self.rng.uniform() < self.pmf(phi)
    
    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate * (1. - terminate)


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1e-2):
        self.rng = rng
        # self.weights = np.zeros((nfeatures, nactions))
        self.weights = np.random.rand(nfeatures, nactions)
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


class LearnedOption:
    
    def __init__(self, n_features, n_actions, rng=np.random.RandomState(1337)):
        self.n_states = n_features
        self.n_actions = n_actions
        
        # Each option has its own policy and termination functions learned via policy gradients
        self.policy = SoftmaxPolicy(rng, n_features, n_actions)
        self.beta = SigmoidTermination(rng, n_features)
