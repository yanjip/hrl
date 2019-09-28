from typing import Callable


class Option:
    
    def __init__(self,
                 initiation_set,
                 termination_function: Callable,
                 policy: Callable,
                 name: str):
        self.initiation_set = initiation_set
        self.termination_function = termination_function
        self.policy = policy
        self.name = name
    
    # @property
    # def initiation_set(self):
    #     return self._initiation_set()
    #
    # @initiation_set.setter
    # def initiation_set(self, v):
    #     self._initiation_set = v
    
    # def sample(self, *args, **kwargs):
    #     raise NotImplementedError
    
    def __str__(self):
        return self.name
