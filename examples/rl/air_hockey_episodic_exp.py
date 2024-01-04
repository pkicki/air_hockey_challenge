from copy import copy
import os, sys
from time import perf_counter

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.random
import wandb
from experiment_launcher import run_experiment, single_experiment
from examples.rl.bsmp.bsmp_distribution import DiagonalGaussianBSMPDistribution, DiagonalGaussianBSMPSigmaDistribution

from examples.rl.bsmp.bsmp_eppo import BSMPePPO
from examples.rl.bsmp.bsmp_policy import BSMPPolicy
from examples.rl.bsmp.bspline import BSpline
from examples.rl.bsmp.bspline_timeoptimal_approximator import BSplineFastApproximatorAirHockeyWrapper
from examples.rl.bsmp.context_builder import IdentityContextBuilder
from examples.rl.bsmp.network import ConfigurationTimeNetwork, ConfigurationTimeNetworkWrapper, LogSigmaNetworkWrapper

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

from mushroom_rl.core import Logger, Agent, MultiprocessEnvironment
from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer
from mushroom_rl.distributions import DiagonalGaussianTorchDistribution, CholeskyGaussianTorchDistribution

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

@single_experiment
def experiment(env: str = '7dof-hit',
               n_envs: int = 1,
               alg: str = "bsmp_eppo",
               n_epochs: int = 100000,
               n_steps: int = None,
               n_steps_per_fit: int = None,
               n_episodes: int = 32,
               n_episodes_per_fit: int = 32,
               n_eval_episodes: int = 2,

               batch_size: int = 32,
               use_cuda: bool = False,

               interpolation_order: int = -1,
               double_integration: bool = False,
               checkpoint: str = "None",

               debug: bool = False,
               seed: int = 444,
               quiet: bool = True,
               render: bool = True,
               #render: bool = False,
               results_dir: str = './logs',
               **kwargs):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: add parameter regarding the constraint loss stuff
    agent_params = dict(
        alg=alg,
        checkpoint=checkpoint,
        seed=seed,
        n_q_cps=kwargs['n_q_cps'] if 'n_q_cps' in kwargs.keys() else 11,
        n_t_cps=kwargs['n_t_cps'] if 'n_t_cps' in kwargs.keys() else 10,
        n_pts_fixed_begin=3,
        n_pts_fixed_end=0,
        sigma_init=['sigma_init'] if 'sigma_init' in kwargs.keys() else 0.1,
        sigma_eps=['sigma_eps'] if 'sigma_eps' in kwargs.keys() else 1e-2,
        constraint_lr=kwargs['constraint_lr'] if 'constraint_lr' in kwargs.keys() else 1e-2,
        mu_lr=kwargs['mu_lr'] if 'mu_lr' in kwargs.keys() else 5e-5,
        n_epochs_policy=kwargs['n_epochs_policy'] if 'n_epochs_policy' in kwargs.keys() else 32,
        batch_size=batch_size,
        eps_ppo=kwargs['eps_ppo'] if 'eps_ppo' in kwargs.keys() else 2e-2,
        ent_coeff=kwargs['ent_coeff'] if 'ent_coeff' in kwargs.keys() else 0e-2,
    )

    name = f"""ePPO_stillsamepose_nn_lr{agent_params['mu_lr']}_bs{batch_size}_constrlr{agent_params['constraint_lr']}_
               nep{n_episodes}_neppf{n_episodes_per_fit}_neppol{agent_params['n_epochs_policy']}_epsppo{agent_params['eps_ppo']}_
               sigmainit{agent_params['sigma_init']}_ent{agent_params['ent_coeff']}_seed{seed}"""

    results_dir = os.path.join(results_dir, name)

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)

    if use_cuda:
        TorchUtils.set_default_device('cuda')

    #wandb_run = wandb.init(project="air_hockey_challenge", config={}, dir=results_dir, name=name,
    #           group=f'{env}_{alg}_ePPO_NN_diag_fixedsampling', tags=[str(env)])

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )


    env_params = dict(
        debug=debug,
        interpolation_order=interpolation_order,
        moving_init=False,
        horizon=100,
        gamma=1.,
    )

    env, env_info_ = env_builder(env, n_envs, env_params)

    agent_params["n_dim"] = env_info_["robot"]["n_joints"]

    agent = agent_builder(env_info_, agent_params)
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillsamepose_nonn_sigma_lr0.01_bs32_constrlr0.01_nep32_neppf32_neppol4_epsppo0.02_sigmainit0.1_ent0.0_seed444_betapos1em5_betavel1em3_betaacc_1em4/7dof-hit/agent-444.msh")
    ##agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillsamepose_nonn_sigma_lr0.03_bs32_constrlr0.01_nep32_neppf32_neppol4_epsppo0.05_sigmainit0.1_ent0.0_seed444/7dof-hit/agent-444.msh")
    ##agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillsamepose_nonn_lr0.03_bs32_constrlr0.01_nep32_neppf32_neppol4_epsppo0.02_sigmainit0.1_ent0.0_seed444/7dof-hit/agent-444.msh")
    #print("Load agent from: ", agent_path)
    #agent = Agent.load(agent_path)
    #agent.bsmp_agent.load_robot()
    #agent.bsmp_agent.policy._traj_no = 0

    if n_envs > 1:
        core = ChallengeCoreVectorized(agent, env, action_idx=[0, 1])
    else:
        core = ChallengeCore(agent, env, action_idx=[0, 1])

    best_success = -np.inf
    best_J = -np.inf
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,
                   n_episodes=n_episodes, n_episodes_per_fit=n_episodes_per_fit, quiet=quiet)

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
        print("BEST J: ", best_J)
        if hasattr(agent, "get_alphas"):
            wandb.log({
            "alphas": {str(i): a for i, a in enumerate(agent.get_alphas())}
            }, step=epoch)
        #if best_success <= success:
        #    best_success = success
        #    logger.log_agent(agent)
        if best_J <= J:
            best_J = J
            logger.log_agent(agent)

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
    elif alg == "bsmp_eppo":
        bsmp_agent = build_agent_BSMPePPO(env_info, **agent_params)
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


def build_agent_BSMPePPO(env_info, **agent_params):

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
    n_q_pts = agent_params["n_q_cps"]
    n_t_pts = agent_params["n_t_cps"]
    n_pts_fixed_begin = agent_params["n_pts_fixed_begin"]
    n_pts_fixed_end = agent_params["n_pts_fixed_end"]
    n_dim = agent_params["n_dim"]
    n_trainable_q_pts = n_q_pts - (n_pts_fixed_begin + n_pts_fixed_end)
    n_trainable_t_pts = n_t_pts
    n_trainable_pts = n_dim * n_trainable_q_pts + n_trainable_t_pts

    q_bsp = BSpline(n_q_pts)
    t_bsp = BSpline(n_t_pts)

    mdp_info = env_info['rl_info']

    #sigma = torch.tensor([0.0650, 0.0650, 0.0650, 0.0651, 0.0652, 0.0653, 0.0649, 0.0652, 0.0650,
    #    0.0650, 0.0648, 0.0651, 0.0652, 0.0649, 0.0651, 0.0651, 0.0650, 0.0650,
    #    0.0652, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0651, 0.0651, 0.0650,
    #    0.0649, 0.0649, 0.0648, 0.0650, 0.0649, 0.0650, 0.0649, 0.0649, 4.9652,
    #    4.9648, 4.9650, 4.9655, 4.9658, 4.9649, 4.9656, 0.4653, 0.4651, 0.4656,
    #    0.4658, 0.4654, 0.4655, 0.4654, 0.0650, 0.0649, 0.0651, 0.0650, 0.0650,
    #    0.0653, 0.0650, 0.1344, 0.0788, 0.0670, 0.0820, 0.0676, 0.0667, 0.0682,
    #    0.0714, 0.0654, 0.0709])
    sigma = torch.tensor([0.0237, 0.0236, 0.0238, 0.0238, 0.0238, 0.0240, 0.0238, 0.0237, 0.0237,
        0.0236, 0.0236, 0.0237, 0.0238, 0.0235, 0.0237, 0.0237, 0.0235, 0.0237,
        0.0238, 0.0236, 0.0236, 0.0237, 0.0238, 0.0236, 0.0238, 0.0238, 0.0237,
        0.0236, 0.0236, 0.0236, 0.0238, 0.0237, 0.0238, 0.0236, 0.0237, 1.5886,
        1.6154, 1.6018, 1.6140, 1.6146, 1.5856, 1.5870, 0.1880, 0.1880, 0.1936,
        0.1866, 0.1864, 0.1842, 0.1842, 0.0333, 0.0373, 0.0362, 0.0351, 0.0318,
        0.0335, 0.0335, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
        0.1000, 0.0853, 0.1089])

    mu_approximator = Regressor(TorchApproximator,
                                network=ConfigurationTimeNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts, n_trainable_t_pts))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=LogSigmaNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts, n_trainable_t_pts))

    mu = torch.zeros(n_trainable_pts)
    policy = BSMPPolicy(env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end, robot_constraints)
    #sigma = agent_params["sigma_init"] * torch.ones(n_trainable_pts)
    #dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator)
    #dist = DiagonalGaussianBSMPDistribution(mu_approximator, sigma)
    #sigma = agent_params["sigma_init"]**2 * torch.eye(n_trainable_pts)
    #sigma_q_init = agent_params["sigma_init"] / 10.
    #sigma_q = sigma_q_init * np.ones((n_trainable_q_pts, n_dim))
    ##sigma_q[-1] /= 2.
    #sigma_t_init = agent_params["sigma_init"]
    #sigma_t = sigma_t_init * np.ones((n_trainable_t_pts))
    ##sigma_t[0] /= 2.
    ##sigma_t[-1] /= 2.

    #sigma = torch.tensor(np.concatenate([sigma_q.reshape(-1), sigma_t]), dtype=torch.float32)
    #sigma = torch.diag(sigma**2)
    #dist = CholeskyGaussianTorchDistribution(mu, sigma)

    #dist = DiagonalGaussianTorchDistribution(mu, sigma)
    #dist = DiagonalGaussianBSMPDistribution(mu_approximator, sigma)
    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator)


    #sigma_optimizer = AdaptiveOptimizer(eps=0.3)
    #mu_optimizer = torch.optim.Adam(self.mu_approximator.model.network.parameters(), lr=agent_params["mu_lr"])
    optimizer = {'class': optim.Adam,
                 'params': {'lr': agent_params["mu_lr"],
                            'weight_decay': 0.0}}

    context_builder = None
    if dist.is_contextual:
        context_builder = IdentityContextBuilder()

    eppo_params = dict(n_epochs_policy=agent_params["n_epochs_policy"],
                       batch_size=agent_params["batch_size"],
                       eps_ppo=agent_params["sigma_eps"],
                       ent_coeff=agent_params["ent_coeff"],
                       context_builder=context_builder
                       )

    agent = BSMPePPO(mdp_info, dist, policy, optimizer, robot_constraints,
                     agent_params["constraint_lr"], **eppo_params)
    return agent

def compute_metrics(core, eval_params):
    print("EVAL")
    #with torch.no_grad():
    #    tmp = core.agent.bsmp_agent.distribution._log_sigma.data.detach().clone()
    #    core.agent.bsmp_agent.distribution._log_sigma.copy_(-1e1 * torch.ones_like(tmp))
    #    dataset = core.evaluate(**eval_params)
    #    core.agent.bsmp_agent.distribution._log_sigma.copy_(tmp)
    #core.agent.bsmp_agent.distribution._evaluate = True
    with torch.no_grad():
        dist = copy(core.agent.bsmp_agent.distribution)
        #sigma = 1e-8 * torch.eye(dist._mu.shape[0])
        #dist_eval = CholeskyGaussianTorchDistribution(dist._mu, sigma)
        #sigma = 1e-8 * torch.ones(dist._mu.shape[0])
        #dist_eval = DiagonalGaussianTorchDistribution(dist._mu, sigma)
        policy = core.agent.bsmp_agent.policy
        sigma_shape = policy._n_trainable_q_pts * policy.n_dim + policy._n_trainable_t_pts
        sigma = 1e-8 * torch.ones(sigma_shape)
        dist_eval = DiagonalGaussianBSMPDistribution(dist._mu_approximator, sigma)
        core.agent.bsmp_agent.distribution = dist_eval
        dataset = core.evaluate(**eval_params)
        core.agent.bsmp_agent.distribution = dist
    #core.agent.bsmp_agent.distribution._evaluate = False

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
