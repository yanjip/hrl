from pathlib import Path

from gym_minigrid.wrappers import FullyObsWrapper

from hrl.envs.four_rooms import FourRooms
from hrl.experiments import EXPERIMENT_DIR
from hrl.learning_algorithms.SMDP import SMDPModelLearning, SMDPPlanning
from hrl.frameworks.options.hard_coded_options import HallwayOption, PrimitiveOption
from hrl.project_logger import ProjectLogger
from hrl.utils import cache
from hrl.visualization.plotter_one_hot import PlotterOneHot

""" Evaluate the benefits of planning with options. """

SAVEPATH = Path(f'{EXPERIMENT_DIR}/SMDP_planning')

if __name__ == '__main__':
    
    # Create environment
    env = FullyObsWrapper(FourRooms(goal_pos=(15, 15)))
    
    # Create loggers
    LOGLEVEL = 20
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
    
    logger.info('Learning option models')
    agent = SMDPModelLearning(env, options=options, loglevel=LOGLEVEL)
    _, R, P = cache(cachedir=SAVEPATH)(agent.get_true_models)(seed=1)
    
    if LOGLEVEL == 10:
        logger.info('Visualizing dynamics models')
        plot_path = SAVEPATH / 'P_models'
        plot_path.mkdir(parents=True, exist_ok=True)
        for option, model in zip(options, P):
            option = option.name
            plotter.plot_dynamics_model(option, model, plot_path / option)
        
        logger.info('Visualizing reward models')
        plot_path = SAVEPATH / 'R_models'
        plot_path.mkdir(parents=True, exist_ok=True)
        for option, model in zip(options, R):
            option = option.name
            plotter.plot_reward_model(option, model, plot_path / option)
    
    logger.info('Planning with options')
    plot_path = SAVEPATH / 'planning_with_options'
    plot_path.mkdir(parents=True, exist_ok=True)
    agent = SMDPPlanning(env, R, P, loglevel=LOGLEVEL)
    for i, v in enumerate(agent.svi(θ=1)):
        plotter.plot_option_value_function(
            option_name='μ',
            option_state_values=v,
            save_path=plot_path / f'μ_svi_{i}'
        )
    logger.info(f'Planning took {i} iterations')
    
    logger.info('Planning without options')
    plot_path = SAVEPATH / 'planning_without_options'
    plot_path.mkdir(parents=True, exist_ok=True)
    ix = len(PrimitiveOption.primitive_options)
    agent = SMDPPlanning(env, R[-ix:], P[-ix:], loglevel=LOGLEVEL)
    for i, v in enumerate(agent.svi(θ=1)):
        plotter.plot_option_value_function(
            option_name='μ',
            option_state_values=v,
            save_path=plot_path / f'μ_svi_{i}'
        )
    logger.info(f'Planning took {i} iterations')
    
    logger.critical(f'Experiment results are in {SAVEPATH}')
