import numpy as np


class Option:
    # TODO: deprecate and use GVF class instead
    def __init__(self,
                 termination_set: np.ndarray,
                 policy: np.ndarray,
                 initiation_set: np.ndarray = None,
                 name: str = None):
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


class MarkovOption:
    
    def __init__(self,
                 option,
                 k: int = 0,
                 starting_state=None,
                 cumulant=0,
                 ):
        """ An option object that is currently being executed by the agent

        :param k: duration of the option so far
        :param starting_state: state in which the option was initiated
        :param cumulant: accumulated signal so far
        """
        self.k = k
        self.starting_state = starting_state
        self.cumulant = cumulant
        self.option = option
    
    def __getattribute__(self, name):
        try:
            return getattr(object.__getattribute__(self, 'option'), name)
        except AttributeError:
            return self.__dict__[name]
    
    def __str__(self):
        return getattr(object.__getattribute__(self, 'option'), 'id')
    
    def reset(self):
        self.k = 0
        self.starting_state = None
        self.cumulant = 0
