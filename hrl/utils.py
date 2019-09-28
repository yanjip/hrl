from queue import Queue
from threading import Thread
import _pickle
import sys
import os
import numpy as np
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def randargmax(array):
    """ a random tie-breaking argmax for 1 dimensional array"""
    return np.random.choice(np.flatnonzero(array == array.max()))
    # return np.argmax(np.random.random(ndarray.shape) * (ndarray == ndarray.max()))


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
