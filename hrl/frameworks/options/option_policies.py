import numpy as np
from scipy.special import logsumexp

from hrl.utils import LearningRate


class OptionPolicy:
    """ Base class for defining internal policies for options. """
    
    def __init__(self,
                 nfeatures: int,
                 nactions: int,
                 rng: np.random.RandomState,
                 lr: LearningRate,
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
    
    def value(self, φ: np.ndarray, action: int = None):
        """ Returns a vector of action values for the current state """
        q = np.dot(φ, self.weights)
        if action is None:
            return q
        return q[action]
    
    def pmf(self, φ) -> np.ndarray:
        """ Log-likelihood ratio estimator """
        assert isinstance(φ, np.ndarray)
        v = self.value(φ) / self.temp
        return np.exp(v - logsumexp(v))
    
    def __call__(self, φ, *args, **kwargs):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(φ)))
    
    def update(self, phi, action, critic):
        actions_pmf = self.pmf(phi)
        feature_index = np.where(phi == 1)[0][0]
        self.weights[feature_index, :] -= self.lr.rate * critic * actions_pmf
        self.weights[feature_index, action] += self.lr.rate * critic
