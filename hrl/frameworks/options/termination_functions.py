import numpy as np
from scipy.special import expit

from hrl.utils import LearningRate


class SigmoidTermination:
    
    def __init__(self, rng, nfeatures, lr: LearningRate):
        self.rng = rng
        # self.weights = np.random.uniform(size=nfeatures)
        self.weights = np.zeros((nfeatures,))
        self.lr = lr
    
    def pmf(self, φ) -> np.ndarray:
        """ Returns probability of continuing in the current state """
        if isinstance(φ, np.ndarray):
            return expit(np.sum(np.dot(self.weights, φ)))
        return expit(np.sum(self.weights[φ]))
    
    def __call__(self, φ, *args, **kwargs):
        """ Decide whether the option should terminate in a given state """
        return self.rng.uniform() < self.pmf(φ)
    
    def update(self, φ, advantage):
        magnitude = self._grad(φ)
        feature_index = np.where(φ == 1)[0][0]
        self.weights[feature_index] -= self.lr.rate * advantage * magnitude
    
    def _grad(self, φ) -> np.ndarray:
        terminate = self.pmf(φ)
        return terminate * (1. - terminate)


class OneStepTermination:
    
    def __call__(self, *args, **kwargs):
        return 1
    
    @staticmethod
    def pmf(φ):
        return 1.
