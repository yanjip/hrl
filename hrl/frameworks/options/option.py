from typing import Callable


class Option:
    def __init__(self,
                 termination: Callable,
                 policy: Callable,
                 initiation: Callable,
                 id: str = None):
        self.I = initiation
        self.β = termination
        self.π = policy
        
        self.id = str(id)
    
    def initiation(self, s, *args, **kwargs):
        return self.I(s, *args, **kwargs)
    
    def termination(self, s, *args, **kwargs):
        return self.β(s, *args, **kwargs)
    
    def policy(self, s, *args, **kwargs):
        return self.π(s, *args, **kwargs)
    
    def __repr__(self):
        return self.id


class MarkovOption:
    
    def __init__(self,
                 option: Option,
                 k: int = 0,
                 starting_state=None,
                 cumulant=0):
        """ An option object that is currently being executed by the agent
        
        :param option: an plain option to wrap the trackers around
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
    
    def __repr__(self):
        return self.option
    
    def reset(self):
        self.k = 0
        self.starting_state = None
        self.cumulant = 0
