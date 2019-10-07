import random
from typing import Callable

import numpy as np

Value = np.float64
ActionDistribution = np.ndarray
Features = np.ndarray


class GVF:
    """ Base class for a General Value Function """
    
    def __init__(self,
                 target_policy: Callable,
                 termination: Callable,
                 cumulant: Callable,
                 eligibility: Callable,
                 feature: Callable,
                 behavioural_policy: Callable,
                 weights: np.ndarray,
                 id=random.randint(0, 10000),
                 *args, **kwargs):
        # Questions
        self.π = target_policy
        self.γ = termination
        self.z = cumulant
        
        # Answers
        self.λ = eligibility
        self.φ = feature
        self.μ = behavioural_policy
        
        self.id = str(id)
        self.w = weights
    
    def initiation_set(self, s) -> bool:
        return True
    
    def __str__(self):
        return self.id
    
    def __repr__(self):
        return self.__class__.__name__
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    def predict(self, s) -> Value:
        """ Specifies the expected value of the target,
        given actions are generated according to the target policy """
        return np.dot(self.w, self.feature(s))
    
    def update(self, *args, **kwargs):
        raise NotImplementedError
    
    def target_policy(self, s) -> ActionDistribution:
        return self.π(s)
    
    def termination(self, s) -> float:
        """ Outputs termination signal based on the agent’s observation of the MDP state """
        return self.γ(s)
    
    def cumulant(self, s) -> float:
        """ Accumulates future values of the signal """
        return self.z(s)
    
    def feature(self, s) -> Features:
        """ A mapping from MDP states to features """
        return self.φ(s)
    
    def behavioural_policy(self, s) -> ActionDistribution:
        return self.μ(s)
    
    def eligibility(self, s) -> Value:
        return self.λ(s)


class GAVF(GVF):
    """ Base class for General Action-Value Function """
    
    def predict(self, s, a) -> Value:
        return np.dot(self.w, self.feature(s, a))
    
    def feature(self, s, a) -> Features:
        return self.φ(s, a)


class PredictionDemon:
    pass


class ControlDemon:
    pass
