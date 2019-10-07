import gym
import numpy as np
from tqdm import tqdm

from hrl.envs.four_rooms import FourRooms
from hrl.envs.wrappers import SimplifyActionSpace, OneHotObsWrapper
from hrl.frameworks.gvf import GAVF
from hrl.frameworks.options import SigmoidTermination, SoftmaxPolicy, PrimitivePolicy, OneStepTermination
from hrl.frameworks.options.policies import EgreedyPolicy
from hrl.learning_algorithms.intra_option import IntraOptionQLearning
from hrl.learning_algorithms.option_critic import OptionCritic
from hrl.project_logger import ProjectLogger
from hrl.utils import LearningRate
from hrl.visualization.plotter_one_hot import PlotterOneHot


def create_options(env: gym.Env, n: int, rng: np.random.RandomState):
    obs = env.reset()
    n_options = n
    n_actions = 3
    n_states = np.prod(obs.shape)
    n_features = n_states * n_actions
    
    α_θ = LearningRate(start_rate=0.25, min_rate=0.20, decay=0.25 / 5000)
    α_υ = LearningRate(start_rate=0.25, min_rate=0.20, decay=0.25 / 5000)
    
    options = list()
    
    class Feature:
        def __init__(self, input_dim: tuple, output_dim: int):
            self.input_dim = input_dim
            self.output_dim = output_dim
        
        def __call__(self, state, action, *args, **kwargs):
            """ Assumes state is one hot and action is an int """
            ix = np.where(state == 1)[0][0]
            return np.eye(self.output_dim)[ix * self.input_dim[1] + action]
    
    # # Primitive options
    # for i in range(n_actions):
    #     target_policy = PrimitivePolicy(action=i, nactions=n_actions)
    #     options.append(GAVF(
    #         target_policy=target_policy,
    #         termination=OneStepTermination(),
    #         cumulant=lambda: 1,
    #         eligibility=lambda: 0,
    #         feature=lambda x: x,
    #         behavioural_policy=target_policy,
    #         weights=np.zeros(1),
    #         id=i,
    #     ))
    
    # Learned options
    for i in range(n_options):
        target_policy = SoftmaxPolicy(nfeatures=n_states, nactions=n_actions,
                                      lr=α_θ, rng=rng)
        options.append(GAVF(
            target_policy=target_policy,
            termination=SigmoidTermination(rng, n_states, α_υ),
            cumulant=lambda: 1,
            eligibility=lambda: 0,
            feature=Feature((n_states, n_actions), n_features),
            behavioural_policy=target_policy,
            weights=np.zeros(1),
            id=i,
        ))
    
    return options


if __name__ == '__main__':
    # Create environment
    tasks = iter([(15, 15), (10, 17), (17, 10)])
    env = OneHotObsWrapper(SimplifyActionSpace(
        FourRooms(agent_pos=(1, 1), goal_pos=next(tasks))))
    env.unwrapped.max_steps = 1000000
    # env.step = partial(stochastic_step, env)
    
    # Set up loggers
    loglevel = 10
    logger = ProjectLogger(level=loglevel, printing=False)
    plotter = PlotterOneHot(env)
    # db = redis.StrictRedis(port=6379)
    
    # Create options
    rng = np.random.RandomState(1338)
    n = 8
    options = create_options(env, n=n, rng=rng)
    
    # Define actors
    actor = EgreedyPolicy(ε=0.02, rng=rng, options=options, loglevel=20)
    
    # Define critics
    α = LearningRate(start_rate=0.5, min_rate=0.4, decay=0.5 / 10000)


    class Feature:
        def __init__(self, input_dim: tuple, output_dim: int):
            self.input_dim = input_dim
            self.output_dim = output_dim
    
        def __call__(self, state, action, *args, **kwargs):
            """ Assumes state is one hot and action is an int """
            ix = np.where(state == 1)[0][0]
            phi = np.zeros(self.output_dim)
            phi[ix * self.input_dim[1] + action] = 1
            return phi


    critic = IntraOptionQLearning(
        env=env, options=options, target_policy=actor, lr=α, loglevel=loglevel,
        feature_generator=Feature((1444, n), 1444*n))
    # Instantiate learning agent
    agent = OptionCritic(
        env=env,
        critic=critic,
        actor=actor,
        action_critic=None,
        loglevel=loglevel
    )
    n_episodes = 5000
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    for t in tqdm(range(n_episodes)):
        agent.learn(baseline=True, render=False)
        
        # Update learning rates
        agent.critic.lr.update(t)
        for o in agent.actor.options:
            if not isinstance(o.π, PrimitivePolicy):
                o.π.lr.update(t)
                o.γ.lr.update(t)
        
        # Plot stats
        grid_dim = (4, env.unwrapped.width, env.unwrapped.height)
        all_states = np.eye(int(np.prod(grid_dim)))
        
        # # Retrieve option-values and policy over options
        # p = list()
        # q = list()
        # for state in all_states:
        #     option_values = agent.critic.option_values(state)
        #     option = agent.actor(state, option_values)
        #     p.append(agent.critic.option_idx_dict[str(option)])
        #     q.append(option_values)
        # p = np.array(p).reshape(grid_dim)
        # q = np.mean(np.array(q).T.reshape((n, *grid_dim)), axis=1)
        
        # logger.info('\nPolicy over options', attrs=['reverse'])
        # p = p.astype(str)
        # for i, a in enumerate(plotter.unicode_actions):
        #     p[p == str(i)] = a
        # for policy, d in zip(p, ['right', 'down', 'left', 'up']):
        #     print(f'direction: {d}')
        #     print(policy)
        
        # logger.info('\nOption values', attrs=['reverse'])
        # for i, option in enumerate(q):
        #     print(options[i].π)
        #     print(option)
        
        # Retrieve terminations and polices of options
        terms = np.zeros((len(options), all_states.shape[0]))
        policies = np.zeros_like(terms)
        for o, option in enumerate(agent.critic.options):
            for s, state in enumerate(all_states):
                terms[o, s] = option.γ.pmf(state)
                policies[o, s] = option.target_policy(state)
        terms = np.mean(terms.reshape((len(options), *grid_dim)), axis=1)
        policies = policies.reshape((len(options), *grid_dim))
        
        logger.info('\nTerminations', attrs=['reverse'])
        if t % 50 == 0:
            for i, option in enumerate(terms):
                print(options[i].γ)
                print(option)
        
        # logger.info("\nOptions' policies", attrs=['reverse'])
        # policies = policies.astype(str)
        # for i, a in enumerate(plotter.unicode_actions):
        #     p[p == str(i)] = a
        # for option_p in policies:
        #     for policy, d in zip(p, ['right', 'down', 'left', 'up']):
        #         print(f'direction: {d}')
        #         print(policy)
        
        # db.set('Options-Q-Values', pickle.dumps(Q))
        # # db.set('Options-Qu-Values', pickle.dumps(Qu))
        # db.set('Terminations', pickle.dumps(term))
        # db.set('Policies', pickle.dumps(pi))
