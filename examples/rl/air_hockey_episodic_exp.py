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
               n_envs: int = 2,
               alg: str = "bsmp",
               n_epochs: int = 100000,
               n_steps: int = None,
               n_steps_per_fit: int = None,
               n_episodes: int = 2,
               n_episodes_per_fit: int = 2,
               n_eval_episodes: int = 1,

               batch_size: int = 64,
               use_cuda: bool = False,

               interpolation_order: int = -1,
               double_integration: bool = False,
               checkpoint: str = "None",

               debug: bool = False,
               seed: int = 444,
               quiet: bool = True,
               #render: bool = True,
               render: bool = False,
               results_dir: str = './logs',
               **kwargs):
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)

    if use_cuda:
        TorchUtils.set_default_device('cuda')

    #wandb_run = wandb.init(project="air_hockey_challenge", config={}, dir=results_dir, name=f"seed_{seed}",
    #           group=f'{env}_{alg}_acc-{double_integration}', tags=[str(env), str(slack_beta)])

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )


    env_params = dict(
        debug=debug,
        interpolation_order=interpolation_order,
        moving_init=False,
        horizon=250,
        gamma=1.,
    )

    env, env_info_ = env_builder(env, n_envs, env_params)

    # TODO: add parameter regarding the constraint loss stuff
    agent_params = dict(
        alg=alg,
        checkpoint=checkpoint,
        seed=seed,
        n_q_cps=kwargs['n_q_cps'] if 'n_q_cps' in kwargs.keys() else 11,
        n_t_cps=kwargs['n_t_cps'] if 'n_t_cps' in kwargs.keys() else 10,
        n_dim=env_info_["robot"]["n_joints"],
        n_pts_fixed_begin=2,
        n_pts_fixed_end=0,
        sigma_init=['sigma_init'] if 'sigma_init' in kwargs.keys() else 0.1,
        sigma_eps=['sigma_eps'] if 'sigma_eps' in kwargs.keys() else 1e-2,
        constraint_lr=kwargs['constraint_lr'] if 'constraint_lr' in kwargs.keys() else 1e-2,
        mu_lr=kwargs['mu_lr'] if 'mu_lr' in kwargs.keys() else 5e-5,
    )

    agent = agent_builder(env_info_, agent_params)
    #agent = Agent.load("./logs/0/7dof-hit/agent-0.msh")
    #agent = Agent.load(os.path.join(os.path.dirname(__file__), "trained_models/0/7dof-hit/agent-0.msh"))
    #agent = Agent.load(os.path.join(os.path.dirname(__file__), "trained_models/noee/7dof-hit/agent-0.msh"))
    #agent.bsmp_agent.load_robot()
    #agent.bsmp_agent.q_log_t_cps_sigma_trainable = -1e2 * np.ones_like(agent.bsmp_agent.q_log_t_cps_sigma_trainable)

    if n_envs > 1:
        core = ChallengeCoreVectorized(agent, env, action_idx=[0, 1])
    else:
        core = ChallengeCore(agent, env, action_idx=[0, 1])

    best_success = -np.inf
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,
                   n_episodes=n_episodes, n_episodes_per_fit=n_episodes_per_fit, quiet=quiet)

        if hasattr(agent, "update_alphas"):
            agent.update_alphas()
        # Evaluate
        J, R, success, c_avg, c_max = compute_metrics(core, eval_params)

        if "logger_callback" in kwargs.keys():
            kwargs["logger_callback"](J, R, success, c_avg, c_max)

        # Write logging
        logger.log_numpy(J=J, R=R, success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                         c_max=np.max(np.concatenate(list(c_max.values()))))
        logger.epoch_info(epoch, J=J, R=R, success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                          c_max=np.max(np.concatenate(list(c_max.values()))))
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
        }, step=epoch)
        if hasattr(agent, "get_alphas"):
            wandb.log({
            "alphas": {str(i): a for i, a in enumerate(agent.get_alphas())}
            }, step=epoch)
        if best_success <= success:
            best_success = success
            #logger.log_agent(agent)

    agent = Agent.load(os.path.join(logger.path, f"agent-{seed}.msh"))

    core = ChallengeCore(agent, env, action_idx=[0, 1])

    eval_params["n_episodes"] = 20
    J, R, best_success, c_avg, c_max = compute_metrics(core, eval_params)
    wandb.log(dict(J=J, R=R, best_success=best_success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                   c_max=np.max(np.concatenate(list(c_max.values())))))
    print("Best Success", best_success)
    wandb_run.finish()


def env_builder(env_name, n_envs, env_params):
    # Specify the customize reward function
    if "hit" in env_name:
        env_params["custom_reward_function"] = HitReward()

    if "defend" in env_name:
        env_params["custom_reward_function"] = DefendReward()

    if "prepare" in env_name:
        env_params["custom_reward_function"] = PrepareReward()

    env = AirHockeyChallengeWrapper(env_name, **env_params)
    if n_envs > 1:
        return MultiprocessEnvironment(AirHockeyChallengeWrapper, env_name, n_envs=n_envs, **env_params), env.env_info
    return env, env.env_info


def agent_builder(env_info, agent_params):
    alg = agent_params["alg"]

    # If load agent from a checkpoint
    if agent_params["checkpoint"] != "None":
        checkpoint = agent_params["checkpoint"]
        seed = agent_params["seed"]
        del agent_params["checkpoint"]
        del agent_params["seed"]

        for root, dirs, files in os.walk(checkpoint):
            for name in files:
                if name == f"agent-{seed}.msh":
                    agent_dir = os.path.join(root, name)
                    print("Load agent from: ", agent_dir)
                    agent = Agent.load(agent_dir)
                    return agent
        raise ValueError(f"Unable to find agent-{seed}.msh in {root}")

    if alg == "bsmp":
        bsmp_agent = build_agent_BSMP(env_info, **agent_params)
        return BSMPAgent(env_info, bsmp_agent)


def build_agent_BSMP(env_info, **agent_params):

    table_constraints = env_info["constraints"].get("ee_constr")
    robot_constraints = dict(
        q = env_info["robot"]["joint_pos_limit"][-1],
        q_dot = env_info["robot"]["joint_vel_limit"][-1],
        q_ddot = env_info["robot"]["joint_acc_limit"][-1],
        z_ee = (table_constraints.z_lb + table_constraints.z_ub) / 2.,
        x_ee_lb = table_constraints.x_lb,
        y_ee_lb = table_constraints.y_lb,
        y_ee_ub = table_constraints.y_ub,
    )


    agent = BSMP(env_info['rl_info'], robot_constraints, env_info["dt"], **agent_params)
    return agent


def compute_metrics(core, eval_params):
    dataset = core.evaluate(**eval_params)

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

    return J, R, success, c_avg, c_max


if __name__ == "__main__":
    run_experiment(experiment)
