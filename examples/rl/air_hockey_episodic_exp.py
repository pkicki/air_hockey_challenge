from copy import copy
import os, sys
from time import perf_counter

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.random
import wandb
from experiment_launcher import run_experiment, single_experiment
from examples.rl.bsmp.bsmp_distribution import DiagonalGaussianBSMPDistribution, DiagonalGaussianBSMPSigmaDistribution, DiagonalMultiGaussianBSMPSigmaDistribution

from examples.rl.bsmp.bsmp_eppo import BSMPePPO
from examples.rl.bsmp.bsmp_policy import BSMPPolicy
from examples.rl.bsmp.bspline import BSpline
from examples.rl.bsmp.bspline_timeoptimal_approximator import BSplineFastApproximatorAirHockeyWrapper
from examples.rl.bsmp.context_builder import IdentityContextBuilder
from examples.rl.bsmp.network import ConfigurationTimeNetwork, ConfigurationTimeNetworkWrapper, LogSigmaNetworkWrapper
from examples.rl.bsmp.utils import unpack_data_airhockey
from examples.rl.bsmp.value_network import ValueNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

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
from mushroom_rl.utils.callbacks import CollectDataset
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
        value_lr=kwargs['value_lr'] if 'value_lr' in kwargs.keys() else 5e-4,
        n_epochs_policy=kwargs['n_epochs_policy'] if 'n_epochs_policy' in kwargs.keys() else 32,
        batch_size=batch_size,
        eps_ppo=kwargs['eps_ppo'] if 'eps_ppo' in kwargs.keys() else 2e-2,
        ent_coeff=kwargs['ent_coeff'] if 'ent_coeff' in kwargs.keys() else 0e-3,
        target_entropy=kwargs["target_entropy"] if 'target_entropy' in kwargs.keys() else -50.,
        entropy_lr=kwargs["entropy_lr"] if 'entropy_lr' in kwargs.keys() else 1e-3,
        initial_entropy_bonus=kwargs["initial_entropy_bonus"] if 'initial_entropy_bonus' in kwargs.keys() else 3e-3,
    )

    name = f"""ePPO_stillrandompose_initsigma02nn_nobigsigma_nn512_SACentropy_tarm50lr1em3init3em3_lr{agent_params['mu_lr']}_valuelr{agent_params['value_lr']}_bs{batch_size}_constrlr{agent_params['constraint_lr']}_nep{n_episodes}_neppf{n_episodes_per_fit}_neppol{agent_params['n_epochs_policy']}_epsppo{agent_params['eps_ppo']}_sigmainit{agent_params['sigma_init']}_ent{agent_params['ent_coeff']}_nqcps{agent_params['n_q_cps']}_ntcps{agent_params['n_t_cps']}_seed{seed}"""

    results_dir = os.path.join(results_dir, name)

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)

    if use_cuda:
        TorchUtils.set_default_device('cuda')

    wandb_run = wandb.init(project="air_hockey_challenge_fullrange_still", config={}, dir=results_dir, name=name,
              group=f'test_{env}_{alg}_ePPO_newreward_undiscounted', tags=[str(env)])

    eval_params = dict(
        n_episodes=n_eval_episodes,
        quiet=quiet,
        render=render
    )


    env_params = dict(
        debug=debug,
        interpolation_order=interpolation_order,
        moving_init=False,
        horizon=150,
        gamma=1.0,
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

    dataset_callback = CollectDataset()
    if n_envs > 1:
        core = ChallengeCoreVectorized(agent, env, action_idx=[0, 1], callbacks_fit=[dataset_callback])
    else:
        core = ChallengeCore(agent, env, action_idx=[0, 1], callbacks_fit=[dataset_callback])

    best_success = -np.inf
    best_J_det = -np.inf
    best_J_sto = -np.inf
    #if_learn = False
    if_learn = True
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        if if_learn:
            #core.agent.bsmp_agent.set_deterministic(True)
            core.learn(n_episodes=n_episodes, n_episodes_per_fit=n_episodes_per_fit, quiet=quiet)
            #core.agent.bsmp_agent.set_deterministic(False)
            J_sto = np.mean(dataset_callback.get().discounted_return)
            init_states = dataset_callback.get().get_init_states()
            context = core.agent.bsmp_agent._context_builder(init_states)
            V_sto = np.mean(core.agent.bsmp_agent.value_function(context).detach().numpy())
            E = np.mean(core.agent.bsmp_agent.distribution.entropy(context).detach().numpy())
            VJ_bias = V_sto - J_sto
            constraints_violation_sto = core.agent.bsmp_agent.compute_constraint_losses(torch.cat(dataset_callback.get().theta_list, axis=0), context).detach().numpy()
            constraints_violation_sto_mean = np.mean(constraints_violation_sto, axis=0)
            constraints_violation_sto_max = np.max(constraints_violation_sto, axis=0)
            mu = core.agent.bsmp_agent.distribution.estimate_mu(context)
            constraints_violation_det = core.agent.bsmp_agent.compute_constraint_losses(mu, context).detach().numpy()
            constraints_violation_det_mean = np.mean(constraints_violation_det, axis=0)
            constraints_violation_det_max = np.max(constraints_violation_det, axis=0)
            q, q_dot, q_ddot, t, dt, duration = core.agent.bsmp_agent.policy.compute_trajectory_from_theta(mu, context)
            mean_duration = np.mean(duration.detach().numpy())
            dataset_callback.clean()
        else:
            J_sto = 0.
            V_sto = 0.
            E = 0.
            VJ_bias = 0.
            constraints_violation_sto_mean = np.zeros(18)
            constraints_violation_sto_max = np.zeros(18)
            constraints_violation_det_mean = np.zeros(18)
            constraints_violation_det_max = np.zeros(18)

        # Evaluate
        J_det, R, success, c_avg, c_max, states, actions = compute_metrics(core, eval_params)
        #assert False
        #wandb_plotting(core, states, actions, epoch)

        if "logger_callback" in kwargs.keys():
            kwargs["logger_callback"](J_det, J_sto, V_sto, R, E, success, c_avg, c_max)

        # Write logging
        logger.log_numpy(J_det=J_det, J_sto=J_sto, V_sto=V_sto, VJ_bias=VJ_bias, R=R, E=E,
                         success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                         c_max=np.max(np.concatenate(list(c_max.values()))))
        logger.epoch_info(epoch, J_det=J_det, V_sto=V_sto, VJ_bias=VJ_bias, R=R, E=E,
                          success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                          c_max=np.max(np.concatenate(list(c_max.values()))))
        wandb.log({
            "Reward/": {"J_det": J_det, "J_sto": J_sto, "V_sto": V_sto, "VJ_bias": VJ_bias, "R": R, "success": success},
            "Entropy/": {"E": E, "entropy_bonus": core.agent.bsmp_agent._log_entropy_bonus.exp().detach().numpy()},
            "Constraints_sto/": {"avg/": {str(i): a for i, a in enumerate(constraints_violation_sto_mean)},
                                 "max/": {str(i): a for i, a in enumerate(constraints_violation_sto_max)}},
            "Constraints_det/": {"avg/": {str(i): a for i, a in enumerate(constraints_violation_det_mean)},
                                 "max/": {str(i): a for i, a in enumerate(constraints_violation_det_max)}},
            "Stats/": {"mean_duration": mean_duration},
            # "Constraint": {
            #     "max": {"pos": np.max(c_max['joint_pos_constr']),
            #             "vel": np.max(c_max['joint_vel_constr']),
            #             "ee": np.max(c_max['ee_constr']),
            #             },
            #     "avg": {"pos": np.mean(c_avg['joint_pos_constr']),
            #             "vel": np.mean(c_avg['joint_vel_constr']),
            #             "ee": np.mean(c_avg['ee_constr']),
            #             }
            #},
        }, step=epoch)
        logger.info(f"BEST J_det: {best_J_det}")
        logger.info(f"BEST J_sto: {best_J_sto}")
        if hasattr(agent, "get_alphas"):
            wandb.log({
            "alphas/": {str(i): a for i, a in enumerate(agent.get_alphas())}
            }, step=epoch)
        #if best_success <= success:
        #    best_success = success
        #    logger.log_agent(agent)

        if best_J_det <= J_det:
            best_J_det = J_det
            logger.log_agent(agent, epoch=epoch)

        if best_J_sto <= J_sto:
            best_J_sto = J_sto
            logger.log_agent(agent, epoch=epoch)

    #agent = Agent.load(os.path.join(logger.path, f"agent-{seed}.msh"))

    #core = ChallengeCore(agent, env, action_idx=[0, 1])

    #eval_params["n_episodes"] = 20
    #J, R, best_success, c_avg, c_max = compute_metrics(core, eval_params)
    #wandb.log(dict(J=J, R=R, best_success=best_success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
    #               c_max=np.max(np.concatenate(list(c_max.values())))))
    #print("Best Success", best_success)
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

    #sigma =  torch.tensor([-0.0065, -0.0068, -0.0066, -0.0067, -0.0064, -0.0068, -0.0066,  0.0138,
    #     0.0098,  0.0134,  0.0136,  0.0134,  0.0114,  0.0138,  0.0121,  0.0090,
    #     0.0073,  0.0129,  0.0068,  0.0069,  0.0140,  0.0085,  0.0096,  0.0085,
    #     0.0087,  0.0084,  0.0087,  0.0085,  0.0089,  0.0102,  0.0090,  0.0096,
    #     0.0101,  0.0098,  0.0090,  0.0088,  0.0091,  0.0086,  0.0103,  0.0083,
    #     0.0111,  0.0092,  0.0062,  0.0073,  0.0079,  0.0082,  0.0076,  0.0113,
    #     0.0119,  0.0132,  0.0106,  0.0132, -0.0020,  0.0129,  0.0049,  0.0138,
    #    -0.0064, -0.0070, -0.0067, -0.0073, -0.0063, -0.0072, -0.0066,  0.0327,
    #     0.0319,  0.0326,  0.0314,  0.0326,  0.0322,  0.0329,  0.0042, -0.0067,
    #     0.0016, -0.0077,  0.0027, -0.0070,  0.0073, -0.0037, -0.0053, -0.0021,
    #    -0.0060, -0.0028, -0.0065, -0.0047,  0.0860,  0.0877,  0.0854,  0.0813,
    #     0.0956,  0.0946,  0.0785,  0.0862,  0.0861,  0.0848]).type(torch.FloatTensor).abs()

    #sigma_q = 0.01 * torch.ones((n_trainable_q_pts, n_dim))
    #sigma_q[-2] = 0.05
    #sigma_q[-3] = 0.4
    #sigma_t = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    #sigma = torch.cat([sigma_q.reshape(-1), sigma_t]).type(torch.FloatTensor)

    #sigma = torch.tensor([0.0311, 0.0317, 0.0311, 0.0315, 0.0310, 0.0316, 0.0311, 0.0311, 0.0318,
    #    0.0311, 0.0315, 0.0312, 0.0318, 0.0311, 0.0311, 0.0316, 0.0311, 0.0314,
    #    0.0312, 0.0315, 0.0311, 0.0311, 0.0317, 0.0311, 0.0317, 0.0311, 0.0319,
    #    0.0310, 0.0310, 0.0316, 0.0310, 0.0314, 0.0311, 0.0314, 0.0310, 0.0309,
    #    0.0313, 0.0310, 0.0315, 0.0309, 0.0310, 0.0308, 0.0100, 0.0100, 0.0100,
    #    0.0100, 0.0100, 0.0100, 0.0310, 0.0310, 0.0315, 0.0309, 0.0317, 0.0308,
    #    0.0309, 0.0311, 0.1208, 0.1205, 0.1209, 0.1209, 0.1209, 0.1212, 0.1210,
    #    0.1208, 0.1199, 0.1205]).type(torch.FloatTensor)

    sigma = torch.tensor([0.1238, 0.0915, 0.1241, 0.1216, 0.1237, 0.0924, 0.1237, 0.1308, 0.1303,
        0.1310, 0.1314, 0.1310, 0.1305, 0.1306, 0.1306, 0.1314, 0.1306, 0.1309,
        0.1308, 0.1316, 0.1303, 0.1300, 0.1314, 0.1303, 0.1312, 0.1299, 0.1305,
        0.1299, 0.1209, 0.1186, 0.1221, 0.1096, 0.1206, 0.0469, 0.1204, 0.1184,
        0.1171, 0.1179, 0.0372, 0.1005, 0.0493, 0.0994, 0.0300, 0.0300, 0.0300,
        0.0300, 0.0300, 0.0300, 0.1146, 0.1198, 0.0843, 0.1222, 0.0861, 0.1205,
        0.0068, 0.1215, 0.0984, 0.0394, 0.2153, 0.2476, 0.2546, 0.2559, 0.2477,
        0.1281, 0.0432, 0.0671]).type(torch.FloatTensor)

    #sigma = torch.tensor([-0.0191, -0.0185, -0.0189, -0.0164, -0.0190, -0.0167, -0.0174, -0.0174,
    #    -0.0192, -0.0162, -0.0151, -0.0160, -0.0169, -0.0168, -0.0135, -0.0094,
    #    -0.0146, -0.0092, -0.0131, -0.0155, -0.0141, -0.0219, -0.0179, -0.0222,
    #    -0.0188, -0.0190, -0.0166, -0.0189, -0.0179, -0.0186, -0.0191, -0.0175,
    #    -0.0160, -0.0199, -0.0193,  0.0906,  0.0847,  0.0874,  0.0490,  0.0716,
    #     0.0297,  0.0657,  0.0300,  0.0300,  0.0300,  0.0300,  0.0300,  0.0300,
    #     0.0226, -0.0199, -0.0117, -0.0273, -0.0070, -0.0252, -0.0154,  0.0045,
    #     0.0559,  0.0553,  0.0584,  0.0666,  0.0667,  0.0655,  0.0593,  0.0581,
    #     0.0532,  0.0541]).type(torch.FloatTensor).abs()

    #sigma_q = 0.125 * torch.ones((n_trainable_q_pts, n_dim))
    #sigma_t = torch.tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
    #    0.1000, 0.1000, 0.1420, 0.1641])
    #sigma = torch.cat([sigma_q.reshape(-1), sigma_t]).type(torch.FloatTensor)

    #sigma = torch.tensor([0.1245, 0.1246, 0.1232, 0.1249, 0.1240, 0.1252, 0.1243, 0.1215, 0.1221,
    #    0.1216, 0.1217, 0.1214, 0.1223, 0.1217, 0.1217, 0.1221, 0.1211, 0.1218,
    #    0.1214, 0.1220, 0.1217, 0.1219, 0.1220, 0.1213, 0.1221, 0.1218, 0.1214,
    #    0.1222, 0.1224, 0.1226, 0.1222, 0.1224, 0.1223, 0.1217, 0.1224, 0.2284,
    #    0.2493, 0.2607, 0.2672, 0.2700, 0.2391, 0.1390, 0.1438, 0.2439, 0.2233,
    #    0.2178, 0.2721, 0.1565, 0.1367, 0.2177, 0.2342, 0.2219, 0.2281, 0.2139,
    #    0.2219, 0.2356, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
    #    0.1000, 0.1476, 0.0339])
    #sigma = torch.tensor([0.0650, 0.0650, 0.0650, 0.0651, 0.0652, 0.0653, 0.0649, 0.0652, 0.0650,
    #    0.0650, 0.0648, 0.0651, 0.0652, 0.0649, 0.0651, 0.0651, 0.0650, 0.0650,
    #    0.0652, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0651, 0.0651, 0.0650,
    #    0.0649, 0.0649, 0.0648, 0.0650, 0.0649, 0.0650, 0.0649, 0.0649, 0.0650,
    #    0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650,
    #    0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0649, 0.0651, 0.0650, 0.0650,
    #    0.0653, 0.0650, 0.1344, 0.0788, 0.0670, 0.0820, 0.0676, 0.0667, 0.0682,
    #    0.0714, 0.0654, 0.0709])
    #sigma = torch.tensor([0.0650, 0.0650, 0.0650, 0.0651, 0.0652, 0.0653, 0.0649, 0.0652, 0.0650,
    #    0.0650, 0.0648, 0.0651, 0.0652, 0.0649, 0.0651, 0.0651, 0.0650, 0.0650,
    #    0.0652, 0.0650, 0.0650, 0.0650, 0.0650, 0.0650, 0.0651, 0.0651, 0.0650,
    #    0.0649, 0.0649, 0.0648, 0.0650, 0.0649, 0.0650, 0.0649, 0.0649, 4.9652,
    #    4.9648, 4.9650, 4.9655, 4.9658, 4.9649, 4.9656, 0.4653, 0.4651, 0.4656,
    #    0.4658, 0.4654, 0.4655, 0.4654, 0.0650, 0.0649, 0.0651, 0.0650, 0.0650,
    #    0.0653, 0.0650, 0.1344, 0.0788, 0.0670, 0.0820, 0.0676, 0.0667, 0.0682,
    #    0.0714, 0.0654, 0.0709])
    #sigma = torch.tensor([0.0237, 0.0236, 0.0238, 0.0238, 0.0238, 0.0240, 0.0238, 0.0237, 0.0237,
    #    0.0236, 0.0236, 0.0237, 0.0238, 0.0235, 0.0237, 0.0237, 0.0235, 0.0237,
    #    0.0238, 0.0236, 0.0236, 0.0237, 0.0238, 0.0236, 0.0238, 0.0238, 0.0237,
    #    0.0236, 0.0236, 0.0236, 0.0238, 0.0237, 0.0238, 0.0236, 0.0237, 1.5886,
    #    1.6154, 1.6018, 1.6140, 1.6146, 1.5856, 1.5870, 0.1880, 0.1880, 0.1936,
    #    0.1866, 0.1864, 0.1842, 0.1842, 0.0333, 0.0373, 0.0362, 0.0351, 0.0318,
    #    0.0335, 0.0335, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
    #    0.1000, 0.0853, 0.1089])

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

    value_function_approximator = ValueNetwork(mdp_info.observation_space)

    mu = torch.zeros(n_trainable_pts)
    policy = BSMPPolicy(env_info, env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end, robot_constraints)
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
    #entropy_weights = torch.tensor([1e0, 1e0, 1e0, 1e0, 1e0])
    #splits = [(n_trainable_q_pts - 3) * n_dim, (n_trainable_q_pts - 2) * n_dim, (n_trainable_q_pts - 1) * n_dim, n_trainable_q_pts * n_dim]
    #dist = DiagonalMultiGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, entropy_weights, splits)


    #sigma_optimizer = AdaptiveOptimizer(eps=0.3)
    #mu_optimizer = torch.optim.Adam(self.mu_approximator.model.network.parameters(), lr=agent_params["mu_lr"])
    optimizer = {'class': optim.Adam,
                 'params': {'lr': agent_params["mu_lr"],
                            'weight_decay': 0.0}}

    value_function_optimizer = optim.Adam(value_function_approximator.parameters(), lr=agent_params["value_lr"])

    context_builder = None
    if dist.is_contextual:
        context_builder = IdentityContextBuilder()

    eppo_params = dict(n_epochs_policy=agent_params["n_epochs_policy"],
                       batch_size=agent_params["batch_size"],
                       eps_ppo=agent_params["sigma_eps"],
                       target_entropy=agent_params["target_entropy"],
                       entropy_lr=agent_params["entropy_lr"],
                       initial_entropy_bonus=agent_params["initial_entropy_bonus"],
                       #ent_coeff=agent_params["ent_coeff"],
                       context_builder=context_builder
                       )

    agent = BSMPePPO(mdp_info, dist, policy, optimizer, value_function_approximator, value_function_optimizer,
                     robot_constraints,
                     agent_params["constraint_lr"], **eppo_params)
    return agent

def compute_metrics(core, eval_params):
    with torch.no_grad():
        core.agent.bsmp_agent.set_deterministic(True)
        dataset = core.evaluate(**eval_params)
        core.agent.bsmp_agent.set_deterministic(False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    print(dataset.undiscounted_return)

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

    state = dataset.state
    #state = state.reshape(dataset.n_episodes, int(state.shape[0] / dataset.n_episodes), state.shape[1])
    action = dataset.action
    #action = action.reshape(dataset.n_episodes, int(action.shape[0] / dataset.n_episodes), 2, 7).transpose(0, 1, 3, 2)
    return J, R, success, c_avg, c_max, state, action


def wandb_plotting(core, states, actions, step):
    puck, puck_dot, q, _, q_dot, _, _, _, _ = unpack_data_airhockey(states)
    ee_pos, ee_rot = core.agent.bsmp_agent.compute_forward_kinematics(torch.tensor(q), torch.tensor(q_dot))
    ee_pos = ee_pos.detach().numpy()
    ee_rot = ee_rot.detach().numpy()
    q_d = actions[..., 0]
    q_dot_d = actions[..., 1]
    ee_pos_d, ee_rot_d = core.agent.bsmp_agent.compute_forward_kinematics(torch.tensor(q_d), torch.tensor(q_dot_d))
    ee_pos_d = ee_pos_d.detach().numpy()
    ee_rot_d = ee_rot_d.detach().numpy()
    columns = ['actual', 'desired']
    idx = 0
    wandb.log({"plots/XY" : wandb.plot.line_series(
    xs=[ee_pos[idx, :, 0], ee_pos_d[idx, :, 0]],
    ys=[ee_pos[idx, :, 1], ee_pos_d[idx, :, 1]],
    keys=columns,
    title="XY")}, step=step)
    wandb.log({"plots/Z" : wandb.plot.line_series(
    xs=[np.arange(ee_pos.shape[1]), np.arange(ee_pos_d.shape[1])],
    ys=[ee_pos[idx, :, 2], ee_pos_d[idx, :, 2]],
    keys=columns,
    title="Z")}, step=step)
        


if __name__ == "__main__":
    run_experiment(experiment)
