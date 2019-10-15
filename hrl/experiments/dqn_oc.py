import pickle
from typing import NamedTuple
import itertools

import numpy as np
import redis
import torch
from gym_minigrid.wrappers import FullyObsWrapper
from tqdm import tqdm

from hrl import FourRooms
from hrl.agents.intra_option import IntraOptionDeepQLearning
from hrl.agents.option_critic import OptionCriticNetwork
from hrl.envs.wrappers import SimplifyActionSpace, Torch
from hrl.frameworks.options import Option, PrimitivePolicy, OneStepTermination
from hrl.frameworks.options.policies import EgreedyPolicy
from hrl.models.torch.network_bodies import NatureConvBody, FCBody
from hrl.project_logger import ProjectLogger
from hrl.visualization.plotter_one_hot import PlotterOneHot


# TODO: lift the assumption of discrete action space
#   add a stop/reset functionality for training
#   add human intervention for changing rewards
#   figure out RNG and seeds
#   use a dictionary of actual option names instead of indicies
#   randomize starting state


def create_options(n: int,
                   n_features: int,
                   n_actions: int,
                   use_primitives: bool = False
                   ):
    options = list()
    
    # Primitive options
    if use_primitives:
        for i in range(n_actions):
            options.append(Option(
                termination=OneStepTermination(),
                policy=PrimitivePolicy(action=i, nactions=n_actions),
                initiation=lambda x: 1,
                id=str(i)
            ))
    
    # Learned options
    params = list()
    for i in range(len(options), len(options) + n):
        ω = Option(
            # termination=SigmoidTerminationTorch(rng, n_features),
            termination=FCBody(n_features, (1,), torch.sigmoid),
            # policy=SoftmaxPolicyTorch(n_features, n_actions),
            policy=FCBody(n_features, (n_actions,), torch.nn.functional.softmax),
            initiation=lambda x: 1,
            id=str(i)
        )
        options.append(ω)
        params += [ω.β.layers.parameters(), ω.π.layers.parameters()]
    
    return options, params


if __name__ == '__main__':
    
    # Create environment
    tasks = [(15, 15), (10, 17), (17, 10), (1, 8), (8, 1), (15, 5), (5, 15)]
    

    def setup_env(env):
        # ReseedWrapper
        env = Torch(FullyObsWrapper(SimplifyActionSpace(env)))
        # env.step = partial(stochastic_step, env)
        return env
    
    
    env = setup_env(FourRooms(goal_pos=tasks.pop(0)))
    env.unwrapped.max_steps = 1000000
    obs = env.reset()
    n_states = env.observation_space
    n_actions = env.action_space.n + 1
    
    # Set up loggers
    # TODO: use RLlog
    loglevel = 20
    logger = ProjectLogger(level=loglevel, printing=False)
    plotter = PlotterOneHot(env)
    db = redis.StrictRedis(port=6379)
    logger.critical(env)
    
    # Define a network shared across options' policies and terminations,
    # as well as the critic
    net = NatureConvBody(in_channels=3)
    params = [net.parameters()]
    
    # Create options
    rng = np.random.RandomState(1338)
    n_options = 8
    options, options_params = create_options(n_options, net.feature_dim,
                                             env.action_space.n)
    
    # Define a policy over options
    actor = EgreedyPolicy(ε=0.02, rng=rng, options=options, loglevel=20)
    
    # Define a critic
    critic_head = FCBody(net.feature_dim, (n_options,), gate=lambda x: x)
    critic = IntraOptionDeepQLearning(
        env=env, critic=critic_head, options=options, target_policy=actor,
        feature_generator=lambda x: x, loglevel=loglevel, weights=None,
    )
    
    # Instantiate learning agent
    params += options_params + [critic_head.layers.parameters()]
    optimizer = torch.optim.RMSprop(itertools.chain(*params), 0.001)
    agent = OptionCriticNetwork(env, feature_generator=net, critic=critic,
                                actor=actor, optimizer=optimizer,
                                loglevel=loglevel)
    
    
    class Config(NamedTuple):
        entropy_weight: float = 0.01
        η: float = 0.01
        target_network_update_freq: int = 200
        gradient_clip: int = 5
    
    
    n_episodes = 5000
    switch_time = n_episodes // len(tasks)
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    for t in tqdm(range(n_episodes)):
        agent.learn(Config())
        
        # if t % switch_time == 0:
        #     agent.env.unwrapped._goal_default_pos = tasks.pop(0)
        #
        # # Update learning rates
        # agent.critic.lr.update(t)
        # for o in agent.actor.options:
        #     if not isinstance(o.π, PrimitivePolicy):
        #         o.π.lr.update(t)
        #         o.β.lr.update(t)
        #
        # # Plot stats
        # grid_dim = (4, env.unwrapped.width, env.unwrapped.height)
        # all_states = list(range(n_states))
        #
        # # Retrieve option-values and policy over options
        # p = list()
        # q = list()
        # for state in all_states:
        #     option_values = agent.critic(state)
        #     option = agent.actor(state, option_values)
        #     p.append(agent.critic.option_idx_dict[str(option)])
        #     q.append(option_values)
        # p = np.array(p).reshape(grid_dim)
        # q = np.mean(np.array(q).T.reshape((n_options, *grid_dim)), axis=1)
        #
        # # logger.info('\nPolicy over options', attrs=['reverse'])
        # # p = p.astype(str)
        # # for i, a in enumerate(plotter.unicode_actions):
        # #     p[p == str(i)] = a
        # # for policy, d in zip(p, ['right', 'down', 'left', 'up']):
        # #     logger.debug(f'direction: {d}')
        # #     logger.debug(policy)
        #
        # logger.info('\nOption values', attrs=['reverse'])
        # if t % 50 == 0:
        #     for i, option in enumerate(q):
        #         logger.debug(options[i].π)
        #         logger.debug(option)
        # db.set('Option-Values', pickle.dumps(q))
        #
        # # Retrieve terminations and polices of options
        # terms = np.zeros((len(options), len(all_states)))
        # policies = np.zeros_like(terms)
        # for o, option in enumerate(agent.critic.options):
        #     for s, state in enumerate(all_states):
        #         terms[o, s] = option.β.pmf(state)
        #         policies[o, s] = option.π(state)
        # terms = np.mean(terms.reshape((len(options), *grid_dim)), axis=1)
        # policies = policies.reshape((len(options), *grid_dim))
        #
        # logger.debug('\nTerminations', attrs=['reverse'])
        # if t % 50 == 0:
        #     for i, option in enumerate(terms):
        #         logger.debug(options[i].termination)
        #         logger.debug(option)
        # db.set('Terminations', pickle.dumps(terms))
        #
        # logger.debug("\nOptions' policies", attrs=['reverse'])
        # policies = policies.astype(str)
        # for i, a in enumerate(plotter.unicode_actions):
        #     p[p == str(i)] = a
        # for option_p in policies:
        #     for policy, d in zip(p, ['right', 'down', 'left', 'up']):
        #         logger.debug(f'direction: {d}')
        #         logger.debug(policy)
        
        # db.set('Options-Qu-Values', pickle.dumps(Qu))
        # db.set('Policies', pickle.dumps(pi))
