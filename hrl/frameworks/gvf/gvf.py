import random
from typing import Callable, Union

import numpy as np

ActionDistribution = np.ndarray


class GVF:
    """ Base class for a General Value Function.
    
    Specifies the expected value of the target cumulant,
    given actions are generated according to the target policy.
    
    # The value produced is not necessarily scalar, i.e. in case of estimating
    # an action-function(Q) we get a row vector with values corresponding to
    # each possible action.
    """
    
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
    
    def __str__(self):
        return self.id
    
    def __repr__(self):
        return self.__class__.__name__
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    def predict(self, s, *args, **kwargs):
        """ Specifies the expected value of the target,
        given actions are generated according to the target policy """
        raise NotImplementedError
    
    def target_policy(self, s) -> ActionDistribution:
        return self.π(s)
    
    def termination(self, s) -> float:
        """ Outputs termination signal based on the agent’s observation of the MDP state """
        return self.γ(s)
    
    def cumulant(self, s) -> float:
        """ Accumulates future values of the signal """
        return self.z(s)
    
    def feature(self, s, *args, **kwargs):
        """ A mapping from MDP states to features """
        return self.φ(s, *args, **kwargs)
    
    def behavioural_policy(self, s) -> ActionDistribution:
        return self.μ(s)
    
    def eligibility(self, s):
        return self.λ(s)


class PredictionDemon(GVF):
    pass


class ControlDemon(GVF):
    pass


class LinearGVF(GVF):
    """ GVF is linear in feature vector. """
    
    def predict(self, s, w: np.array = None, *args, **kwargs) -> np.float:
        if w is None:
            w = self.w
        return np.dot(w.T, self.feature(s, *args, **kwargs))


class TabularV(LinearGVF):
    """ A special case of a linear GVF for estimating state-value functions.

    The weight matrix shape is (n_states, ). Assumes categorical states.
    """
    
    def __init__(self, n_states: int, n_options: int = 0, *args, **kwargs):
        # TODO: allow for other initializations than 0
        self.feature_dim = (n_states, n_options) if n_options else (n_states,)
        super().__init__(weights=np.zeros(self.feature_dim), *args, **kwargs)
    
    def feature(self, s: int = None, *args, **kwargs) -> np.ndarray:
        """ Converts discrete state variable to one-hot representation """
        if s is not None:
            phi = np.zeros(self.feature_dim)
            phi[s] = 1
        else:
            phi = np.ones(self.feature_dim)
        return phi


class TabularQ(TabularV):
    """ A special case of a linear GVF for estimating option-value functions.
    
    The weight matrix shape is (n_states, n_options).
    Assumes that states and options are categorical.
    """
    
    def predict(self,
                s: int = slice(None),
                o: int = None,
                w: np.array = None, *args,
                **kwargs) -> Union[np.float, np.array]:
        """ Preforms a table lookup to find a value of a state-option tuple """
        if w is None:
            w = self.w
        return w[s] if o is None else w[s, o]
    
    def feature(self, s: int = slice(None), o: int = None, *args,
                **kwargs) -> np.ndarray:
        """ Constructs a binary matrix of size (n_states, n_options).
        
        Maps discrete state-options tuples to entries in the table of zeros
        everywhere except with 1 at:
            cell (s, o)     when both `s` and `o` are provided
            row (s, :)      when only `s` is provided
            column (:, o)   when only `o` is provided
            everywhere      when neither `s` nor `o` are provided
        """
        phi = np.zeros(self.feature_dim)
        if o is None:
            phi[s] = 1
        else:
            phi[s, o] = 1
        return phi
