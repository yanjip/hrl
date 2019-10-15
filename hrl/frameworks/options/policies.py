from typing import List, Union

import numpy as np
import torch

from hrl.frameworks.options import Option
from hrl.project_logger import ProjectLogger
from hrl.utils import randargmax

Action = int


class PolicyOverOptions:
    
    def __init__(self,
                 options: List[Option],
                 rng: np.random.RandomState,
                 loglevel: int):
        self.options = options
        self.option_names_dict = {str(o): o for o in self.options}
        self.option_idx_dict = {name: i for i, name in
                                enumerate(self.option_names_dict)}
        self.rng = rng
        self.logger = ProjectLogger(level=loglevel, printing=False)
    
    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)
    
    def pmf(self, state, *args, **kwargs):
        """ Returns a vector of probabilities over actions """
        raise NotImplementedError
    
    def act(self, state, *args, **kwargs) -> Union[Option, Action]:
        raise NotImplementedError
    
    def _initiation_set_filter(self, state):
        for i, o in enumerate(self.options):
            if o.initiation(state) == 1:
                yield i, o


class RandomPolicy(PolicyOverOptions):
    """ Picks an optimal option at random """
    
    def pmf(self, state, *args, **kwargs):
        probs = [o.initiation(state) for o in self.options]
        return np.array(probs) / np.sum(probs)
    
    def act(self, state, *args, **kwargs):
        """ Picks an option at random """
        options = self._initiation_set_filter(state)
        return self.rng.choice(options)


class EgreedyPolicy(PolicyOverOptions):
    """ Picks the optimal option with probability 1 - ε """
    
    # TODO: add scheduled annealing
    
    def __init__(self, ε: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ε = ε
    
    def pmf(self, state, action_values):
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().numpy().squeeze()
        probs = np.array([o.initiation(state) for o in self.options],
                         dtype=np.float)
        action_values[np.array(np.abs(probs - 1), dtype=bool)] = np.inf
        probs[probs == 1] = self.ε / (probs.sum() - 1)
        probs[randargmax(action_values)] = 1 - self.ε
        assert np.sum(probs) == 1.
        return probs
    
    def act(self,
            state,
            action_values: Union[np.ndarray, torch.Tensor]
            ) -> Union[Option, Action]:
        # Filter options based on their initiation set
        options = list(self._initiation_set_filter(state))
        option_indices = [i for i, o in options]
        # self.logger.debug(
        #     f'Available options in state {state}: {[str(o) for i, o in options]}')
        
        # Pick greedy wrt to value function
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().numpy().squeeze()
        max_q = action_values[option_indices].max()
        
        # Pick at random from the best available options
        valid = set(option_indices) & set(np.where(action_values == max_q)[0])
        optimal_option = self.options[self.rng.choice(list(valid))]
        # self.logger.debug(f'Best option: {optimal_option}')
        
        if self.rng.uniform() < self.ε:
            return self.rng.choice([self.options[i] for i, o in options])
        return optimal_option


class DistributedRepresentation(PolicyOverOptions):
    pass
