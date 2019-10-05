from pathlib import Path

import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper
from tqdm import tqdm

from hrl.envs.four_rooms import FourRooms
from hrl.experiments import EXPERIMENT_DIR
from hrl.frameworks.options.SMDP import SMDPValueLearning
from hrl.frameworks.options.hard_coded_options import HallwayOption, PrimitiveOption
from hrl.frameworks.options.intra_option import IntraOptionValueLearning
from hrl.project_logger import ProjectLogger
from hrl.visualization import PlotterOneHot

SAVEPATH = Path(f'{EXPERIMENT_DIR}/value_learning')

if __name__ == '__main__':
    
    # Create environment
    tasks = iter([(15, 15), (10, 17), (17, 10), (17, 1), (8, 8)])
    env = FullyObsWrapper(FourRooms(goal_pos=next(tasks)))
    env.unwrapped.max_steps = 1000000
    
    # Create loggers
    LOGLEVEL = 10
    logger = ProjectLogger(level=LOGLEVEL, printing=False)
    logger.critical(env)
    plotter = PlotterOneHot(env=env)
    SAVEPATH /= env.unwrapped.__class__.__name__
    SAVEPATH.mkdir(parents=True, exist_ok=True)
    
    # Create hard-coded options
    options = [HallwayOption(o, env.observation_space.shape[::-1]) for o in
               sorted(HallwayOption.hallway_options)]
    options += [PrimitiveOption(o, env.observation_space.shape[::-1]) for o in
                sorted(PrimitiveOption.primitive_options)]
    
    # Learn the optimal value function
    for agent in (IntraOptionValueLearning, SMDPValueLearning):
        agent_name = agent.__class__.__name__
        agent = agent(env, options, loglevel=LOGLEVEL)
        
        step_count = list()
        history = dict()
        transfer = False
        for i, (q_values, steps) in tqdm(
            iterable=enumerate(agent.q_learning(
                n_episodes=1000)
            ),
            desc=np.mean(step_count[-10:]) if step_count else ''
        ):
            step_count.append(steps)
            logger.info(step_count)
            
            if transfer and i % 500 == 0:
                history[i] = q_values.copy()
                agent.env._goal_default_pos = next(tasks)
