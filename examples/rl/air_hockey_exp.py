import os, sys
from time import perf_counter

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.random
import wandb
from experiment_launcher import run_experiment, single_experiment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore
from air_hockey_challenge.framework.challenge_core_vectorized import ChallengeCoreVectorized
from atacom_agent_wrapper import ATACOMAgent, build_ATACOM_Controller
from network import SACActorNetwork, SACCriticNetwork
from rewards import HitReward, DefendReward, PrepareReward
from rl_agent_wrapper import RlAgent
from bsmp_agent_wrapper import BSMPAgent
from bsmp.agent import BSMP
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Logger, Agent, MultiprocessEnvironment
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.frames import LazyFrames
from mushroom_rl.rl_utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.utils.torch import TorchUtils

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

@single_experiment
def experiment(env: str = '7dof-hit',
               alg: str = "atacom-sac",
               n_envs: int = 1,
               n_steps: int = 50000,
               n_epochs: int = 100,
               quiet: bool = True,
               n_steps_per_fit: int = 1,
               render: bool = False,
               n_eval_episodes: int = 10,

               actor_lr: float = 1e-4,
               critic_lr: float = 3e-4,
               n_features: str = "128 128 128",
               batch_size: int = 64,
               initial_replay_size: int = 10000,
               max_replay_size: int = 200000,
               tau: float = 1e-3,
               warmup_transitions: int = 10000,
               lr_alpha: float = 1e-6,
               target_entropy: float = -10,
               use_cuda: bool = False,

               interpolation_order: int = -1,
               double_integration: bool = False,
               checkpoint: str = "None",

               slack_type: str = 'soft_corner',
               slack_beta: float = 4,
               slack_tol: float = 1e-6,

               seed: int = 0,
               results_dir: str = './logs',
               **kwargs):
    np.random.seed(seed)
    torch.manual_seed(seed)

    configs = {}

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)

    if use_cuda:
        TorchUtils.set_default_device('cuda')

    #wandb_run = wandb.init(project="air_hockey_challenge", config=configs, dir=results_dir, name=f"seed_{seed}",
    #           group=f'{env}_{alg}_acc-{double_integration}', tags=[str(env), str(slack_beta)])

    eval_params = {
        "n_episodes": n_eval_episodes,
        "quiet": quiet,
        "render": render
    }

    kwargs['interpolation_order'] = interpolation_order
    kwargs['debug'] = True
    env, env_info_ = env_builder(env, n_envs, kwargs)

    # TODO: calling locals() is a bad idea beacuse some local variable names may overlap
    # with positional arguments of functions called by the agent_builder
    agent = agent_builder(env_info_, locals())

    if n_envs > 1:
        core = ChallengeCoreVectorized(agent, env, action_idx=[0, 1])
    else:
        core = ChallengeCore(agent, env, action_idx=[0, 1])

    best_success = -np.inf

    for epoch in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=quiet)

        # Evaluate
        J, R, success, c_avg, c_max, E, V, alpha = compute_metrics(core, eval_params)

        if "logger_callback" in kwargs.keys():
            kwargs["logger_callback"](J, R, success, c_avg, c_max, E, V)

        # Write logging
        logger.log_numpy(J=J, R=R, success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                         c_max=np.max(np.concatenate(list(c_max.values()))), E=E, V=V)
        logger.epoch_info(epoch, J=J, R=R, success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                          c_max=np.max(np.concatenate(list(c_max.values()))), E=E, V=V)
        wandb.log({
            "Reward": {"J": J, "R": R, "success": success},
            "Constraint": {
                "max": {"pos": np.max(c_max['joint_pos_constr']),
                        "vel": np.max(c_max['joint_vel_constr']),
                        "ee": np.max(c_max['ee_constr']),
                        },
                "avg": {"pos": np.mean(c_avg['joint_pos_constr']),
                        "vel": np.mean(c_avg['joint_vel_constr']),
                        "ee": np.mean(c_avg['ee_constr']),
                        }
            },
            "Training": {
                "E": E, "V": V, "alpha": alpha,
            },
        }, step=epoch)
        if hasattr(agent, "get_alphas"):
            wandb.log({
            "alphas": {str(i): a for i, a in enumerate(agent.get_alphas())}
            }, step=epoch)
        if best_success <= success:
            best_success = success
            logger.log_agent(agent)

    agent = Agent.load(os.path.join(logger.path, f"agent-{seed}.msh"))

    core = ChallengeCore(agent, env, action_idx=[0, 1])

    eval_params["n_episodes"] = 20
    J, R, best_success, c_avg, c_max, E, V, alpha = compute_metrics(core, eval_params)
    wandb.log(dict(J=J, R=R, best_success=best_success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                   c_max=np.max(np.concatenate(list(c_max.values()))), E=E, V=V))
    print("Best Success", best_success)
    wandb_run.finish()


def env_builder(env_name, n_envs, kwargs):
    settings = {}
    keys = ["gamma", "horizon", "debug", "interpolation_order", "moving_init"]

    for key in keys:
        if key in kwargs.keys():
            settings[key] = kwargs[key]
            del kwargs[key]

    # Specify the customize reward function
    if "hit" in env_name:
        settings["custom_reward_function"] = HitReward()

    if "defend" in env_name:
        settings["custom_reward_function"] = DefendReward()

    if "prepare" in env_name:
        settings["custom_reward_function"] = PrepareReward()

    env = AirHockeyChallengeWrapper(env_name, **settings)
    if n_envs > 1:
        return MultiprocessEnvironment(AirHockeyChallengeWrapper, env_name, n_envs=n_envs, **settings), env.env_info
    return env, env.env_info


def agent_builder(env_info, kwargs):
    alg = kwargs["alg"]

    # If load agent from a checkpoint
    if kwargs["checkpoint"] != "None":
        checkpoint = kwargs["checkpoint"]
        seed = kwargs["seed"]
        del kwargs["checkpoint"]
        del kwargs["seed"]

        for root, dirs, files in os.walk(checkpoint):
            for name in files:
                if name == f"agent-{seed}.msh":
                    agent_dir = os.path.join(root, name)
                    print("Load agent from: ", agent_dir)
                    agent = RlAgent.load(agent_dir)
                    return agent
        raise ValueError(f"Unable to find agent-{seed}.msh in {root}")

    if alg == "sac":
        sac_agent = build_agent_SAC(env_info, **kwargs)
        return RlAgent(env_info, kwargs["double_integration"], sac_agent)

    if alg == "atacom-sac":
        sac_agent = build_agent_SAC(env_info, **kwargs)
        atacom = build_ATACOM_Controller(env_info, **kwargs)
        return ATACOMAgent(env_info, kwargs["double_integration"], sac_agent, atacom)


def build_agent_SAC(env_info, alg, actor_lr, critic_lr, n_features, batch_size,
                    initial_replay_size, max_replay_size, tau,
                    warmup_transitions, lr_alpha, target_entropy,
                    double_integration, **kwargs):
    if type(n_features) is str:
        n_features = list(map(int, n_features.split(" ")))

    if double_integration:
        env_info['rl_info'].action_space = Box(*env_info["robot"]["joint_acc_limit"])
    else:
        env_info['rl_info'].action_space = Box(*env_info["robot"]["joint_vel_limit"])

    if alg == "atacom-sac":
        env_info['rl_info'].action_space = Box(-np.ones(env_info['robot']['n_joints']),
                                               np.ones(env_info['robot']['n_joints']))
        obs_low = np.concatenate([env_info['rl_info'].observation_space.low, -np.ones(env_info['robot']['n_joints'])])
        obs_high = np.concatenate([env_info['rl_info'].observation_space.high, np.ones(env_info['robot']['n_joints'])])
        env_info['rl_info'].observation_space = Box(obs_low, obs_high)

    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=env_info["rl_info"].observation_space.shape,
                           output_shape=env_info["rl_info"].action_space.shape,
                           n_features=n_features,
                           )
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=env_info["rl_info"].observation_space.shape,
                              output_shape=env_info["rl_info"].action_space.shape,
                              n_features=n_features,
                              )

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': actor_lr}}
    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(env_info["rl_info"].observation_space.shape[0] +
                                      env_info["rl_info"].action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': critic_lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         output_shape=(1,),
                         )

    alg_params = dict(initial_replay_size=initial_replay_size,
                      max_replay_size=max_replay_size,
                      batch_size=batch_size,
                      warmup_transitions=warmup_transitions,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      critic_fit_params=None,
                      target_entropy=target_entropy,
                      )

    agent = SAC(env_info['rl_info'], actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                actor_optimizer=actor_optimizer, critic_params=critic_params, **alg_params)

    prepro = MinMaxPreprocessor(env_info["rl_info"])
    agent.add_preprocessor(prepro)
    return agent


def compute_V(agent, dataset):
    def get_init_states(dataset):
        pick = True
        x_0 = list()
        for d in dataset:
            if pick:
                if isinstance(d[0], LazyFrames):
                    x_0.append(np.array(d[0]))
                else:
                    x_0.append(d[0])
            pick = d[-1]
        return np.array(x_0)

    Q = list()
    for state in get_init_states(dataset):
        s = np.array([state for i in range(100)])
        a = np.array([agent.draw_action(state)[2] for i in range(100)])
        Q.append(agent.rl_agent._critic_approximator(s, a).mean())
    return np.array(Q).mean()


def normalize_state(parsed_state, agent):
    normalized_state = parsed_state.copy()
    for i in range(len(normalized_state)):
        normalized_state[i] = agent.preprocess(normalized_state[i])
    return normalized_state


def compute_metrics(core, eval_params):
    dataset = core.evaluate(**eval_params)
    parsed_dataset = dataset.parse(to='numpy')

    rl_agent = core.agent.rl_agent

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.reward)

    eps_length = dataset.episodes_length
    success = 0
    current_idx = 0
    for episode_len in eps_length:
        success += dataset.info["success"][current_idx + episode_len - 1]
        current_idx += episode_len
    success /= len(eps_length)

    c_avg = {key: np.zeros_like(value) for key, value in dataset.info["constraints_value"][0].items()}
    c_max = {key: np.zeros_like(value) for key, value in dataset.info["constraints_value"][0].items()}

    for constraint in dataset.info["constraints_value"]:
        for key, value in constraint.items():
            c_avg[key] += value
            idxs = c_max[key] < value
            c_max[key][idxs] = value[idxs]

    N = len(dataset.info["constraints_value"])
    for key in c_avg.keys():
        c_avg[key] /= N

    normalized_state = normalize_state(parsed_dataset.__next__(), core.agent)
    _, log_prob_pi = rl_agent.policy.compute_action_and_log_prob(normalized_state)
    E = -log_prob_pi.mean()

    V = compute_V(core.agent, dataset)

    alpha = rl_agent._alpha

    return J, R, success, c_avg, c_max, E, V, alpha


if __name__ == "__main__":
    run_experiment(experiment)
