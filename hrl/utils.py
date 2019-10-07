import _pickle
import os
import sys
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def randargmax(array, rng: np.random.RandomState = None):
    """ A random tie-breaking argmax for 1 dimensional array """
    choices = np.flatnonzero(array == array.max())
    if rng:
        return rng.choice(choices)
    return np.random.choice(choices)


class LearningRate:
    __slots__ = 'rate', 'min', 'decay'
    
    def __init__(self, start_rate: float, min_rate: float, decay: float):
        self.rate = start_rate
        self.min = min_rate
        self.decay = decay
    
    def update(self, t):
        """ Updates learning rate """
        update = self.rate / (1. + self.decay * t)
        self.rate = max(self.min, update)


class DevNull(object):
    def write(self, arg):
        pass


class NoPrint:
    
    def __init__(self):
        self._stdout = sys.stdout
    
    def __enter__(self):
        sys.stdout = DevNull()
        return
    
    def __exit__(self, *args):
        sys.stdout = self._stdout


def threaded(f, daemon=False):
    def wrapped_f(q, *args, **kwargs):
        ret = f(*args, **kwargs)
        q.put(ret)
    
    def wrap(*args, **kwargs):
        q = Queue()
        t = Thread(target=wrapped_f, args=(q,) + args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t
    
    return wrap


def cache(cachedir=os.path.join(ROOT_DIR, 'cache')):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f'caching {func.__name__} at {cachedir}')
            filename = f'{cachedir}/{func.__name__}_{args}_{kwargs}.pkl' if args or kwargs else f'{cachedir}/{func.__name__}.pkl'
            filename = Path(filename)
            if filename.exists():
                print(
                    f'function {func.__name__} with arguments {args} and kwargs {kwargs} is already cached')
                with open(filename, 'rb') as f:
                    return _pickle.load(f)
            
            filename.parent.mkdir(parents=True, exist_ok=True)
            result = func(*args, **kwargs)
            with open(filename, 'wb') as f:
                _pickle.dump(result, f)
            return result
        
        return wrapper
    
    return decorator
