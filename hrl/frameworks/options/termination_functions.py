import numpy as np
import torch.nn.functional as F
from scipy.special import expit
from torch import nn

from hrl.utils import LearningRate


class SigmoidTermination:
    
    def __init__(self, rng, nfeatures, lr: LearningRate):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))
        self.lr = lr
    
    def pmf(self, s) -> np.ndarray:
        """ Returns probability of continuing in the current state """
        φ = np.zeros(self.weights.shape[0])
        φ[s] = 1
        return expit(np.sum(np.dot(self.weights, φ)))
    
    def __call__(self, φ, *args, **kwargs):
        """ Decide whether the option should terminate in a given state """
        return self.rng.uniform() < self.pmf(φ)
    
    def update(self, s, advantage):
        magnitude = self._grad(s)
        self.weights[s] -= self.lr.rate * advantage * magnitude
    
    def _grad(self, s) -> np.ndarray:
        terminate = self.pmf(s)
        return terminate * (1. - terminate)


class SigmoidTerminationTorch(nn.Module):
    
    def __init__(self, rng, nfeatures: int):
        super().__init__()
        self.rng = rng
        self.fc = nn.Linear(nfeatures, 1)
    
    def forward(self, phi):
        return F.sigmoid(self.fc(phi))
    
    # def __call__(self, φ, *args, **kwargs):
    #     """ Decide whether the option should terminate in a given state """
    #     return self.rng.uniform() < self.forward(φ)


class OneStepTermination:
    
    def __call__(self, *args, **kwargs):
        return 1
    
    @staticmethod
    def pmf(φ):
        return 1.
