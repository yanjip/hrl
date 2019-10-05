import numpy as np


class Option:
    
    def __init__(self,
                 initiation_set,
                 termination_set: np.ndarray,
                 policy: np.ndarray,
                 name: str):
        self._initiation_set = initiation_set
        self._termination_set = termination_set
        self._policy = policy
        self.name = name

    @property
    def initiation_set(self):
        return self._initiation_set

    @initiation_set.setter
    def initiation_set(self, Ι: np.ndarray):
        self._initiation_set = Ι

    @property
    def termination_function(self):
        return lambda s: self._termination_set[s]

    @termination_function.setter
    def termination_function(self, β):
        self._termination_set = β

    @property
    def policy(self):
        return lambda s: int(self._policy[s])

    @policy.setter
    def policy(self, π):
        self._policy = π
    
    def __str__(self):
        return self.name


class LearnedOption(Option):
    """ Option with learned models """
    
    def __init__(self,
                 R: np.ndarray,
                 P: np.ndarray,
                 *args, **kwargs):
        super(Option, self).__init__(*args, **kwargs)
        self.R = R
        self.P = P


class MarkovOption(Option):
    
    def __init__(self,
                 k: int = 0,
                 starting_state=None,
                 cumulant=0,
                 *args, **kwargs):
        """ An option object that is currently being executed by the agent

        :param k: duration of the option so far
        :param starting_state: state in which the option was initiated
        :param cumulant: accumulated signal so far
        """
        super().__init__(*args, **kwargs)
        self.k = k
        self.starting_state = starting_state
        self.cumulant = cumulant
    
    def reset(self):
        self.k = 0
        self.starting_state = None
        self.cumulant = 0
