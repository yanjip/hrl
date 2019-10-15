import numpy as np
import torch.nn.functional as F
from scipy.special import logsumexp
from torch import nn

from hrl.models.torch.network_utils import layer_init
from hrl.utils import LearningRate


class OptionPolicy:
    """ Base class for defining internal policies for options. """
    
    def __init__(self,
                 nfeatures: int,
                 nactions: int,
                 rng: np.random.RandomState,
                 lr: LearningRate = LearningRate(1, 1, 0),
                 *args, **kwargs):
        self.rng = rng
        # self.weights = np.random.uniform(size=(nfeatures, nactions))
        self.weights = np.zeros((nfeatures, nactions))
        self.lr = lr
    
    def __call__(self, φ: np.ndarray, action: int):
        """ Samples an action wrt the current weights """
        raise NotImplementedError
    
    def value(self, φ: np.ndarray, action: int = None):
        raise NotImplementedError
    
    def pmf(self, φ) -> np.ndarray:
        """ Returns action distribution for the current state """
        raise NotImplementedError


class PrimitivePolicy:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]
    
    def __call__(self, phi):
        return self.action
    
    def pmf(self, phi):
        return self.probs


class SoftmaxPolicy(OptionPolicy):
    def __init__(self, temp: float = 1e-2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp
    
    def value(self, s: int, action: int = None):
        """ Returns a vector of action values for the current state """
        φ = np.zeros(self.weights.shape[0])
        φ[s] = 1
        q = np.dot(φ, self.weights)
        if action is None:
            return q
        return q[action]
    
    def pmf(self, s) -> np.ndarray:
        """ Log-likelihood ratio estimator """
        # assert isinstance(φ, np.ndarray)
        v = self.value(s) / self.temp
        return np.exp(v - logsumexp(v))
    
    def __call__(self, s, *args, **kwargs):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(s)))
    
    def update(self, s, action, critic):
        actions_pmf = self.pmf(s)
        self.weights[s, :] -= self.lr.rate * critic * actions_pmf
        self.weights[s, action] += self.lr.rate * critic


class SoftmaxPolicyTorch(nn.Module):
    
    def __init__(self, n_features: int, n_actions: int):
        super().__init__()
        self.layers = layer_init(nn.Linear(n_features, n_actions))
    
    def forward(self, phi):
        pi = self.layers(phi)
        log_pi = F.log_softmax(pi)
        pi = F.softmax(pi)
        return pi, log_pi
