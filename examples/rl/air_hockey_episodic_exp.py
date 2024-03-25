from copy import copy
import os, sys
from time import perf_counter

import numpy as np
import torch.optim as optim
import torch.random
import wandb
from experiment_launcher import run_experiment, single_experiment
from examples.rl.bsmp.bsmp_distribution import DiagonalGaussianBSMPDistribution, DiagonalGaussianBSMPSigmaDistribution, DiagonalMultiGaussianBSMPSigmaDistribution

from examples.rl.bsmp.bsmp_eppo import BSMPePPO
from examples.rl.bsmp.bsmp_policy import BSMPPolicy
from examples.rl.bsmp.bsmp_stopping_policy import BSMPStoppingPolicy
from examples.rl.bsmp.bsmp_unstructured_policy import BSMPUnstructuredPolicy
from examples.rl.bsmp.bspline import BSpline
from examples.rl.bsmp.context_builder import IdentityContextBuilder
from examples.rl.bsmp.network import ConfigurationNetworkWrapper, ConfigurationTimeNetworkWrapper, LogSigmaNetworkWrapper
from examples.rl.bsmp.promp_policy import ProMPPolicy
from examples.rl.bsmp.utils import equality_loss, limit_loss, unpack_data_airhockey
from examples.rl.bsmp.value_network import ValueNetwork

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore
from air_hockey_challenge.framework.challenge_core_vectorized import ChallengeCoreVectorized
from bsmp_agent_wrapper import BSMPAgent

from mushroom_rl.core import Logger, Agent, MultiprocessEnvironment
from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator

import matplotlib.pyplot as plt

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

@single_experiment
def experiment(env: str = '7dof-hit',
               n_envs: int = 1,
               alg: str = "promp",
               #alg: str = "bsmp_eppo_unstructured",
               #alg: str = "bsmp_eppo_return",
               n_epochs: int = 100000,
               n_steps: int = None,
               n_steps_per_fit: int = None,
               n_episodes: int = 16,
               n_episodes_per_fit: int = 16,
               n_eval_episodes: int = 2,

               batch_size: int = 16,
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

    # TODO: add parameter regarding the constraint loss stuff
    agent_params = dict(
        alg=alg,
        checkpoint=checkpoint,
        seed=seed,
        n_dim=7,
        n_q_cps=kwargs['n_q_cps'] if 'n_q_cps' in kwargs.keys() else 11,
        n_t_cps=kwargs['n_t_cps'] if 'n_t_cps' in kwargs.keys() else 10,
        n_pts_fixed_begin=3,
        n_pts_fixed_end=0 if alg == "bsmp_eppo" else 2,
        sigma_init_q=['sigma_init_q'] if 'sigma_init_q' in kwargs.keys() else 1.0,
        sigma_init_t=['sigma_init_t'] if 'sigma_init_t' in kwargs.keys() else 1.0,
        sigma_eps=['sigma_eps'] if 'sigma_eps' in kwargs.keys() else 1e-2,
        constraint_lr=kwargs['constraint_lr'] if 'constraint_lr' in kwargs.keys() else 1e-2,
        mu_lr=kwargs['mu_lr'] if 'mu_lr' in kwargs.keys() else 5e-5,
        value_lr=kwargs['value_lr'] if 'value_lr' in kwargs.keys() else 5e-4,
        n_epochs_policy=kwargs['n_epochs_policy'] if 'n_epochs_policy' in kwargs.keys() else 32,
        batch_size=batch_size,
        eps_ppo=kwargs['eps_ppo'] if 'eps_ppo' in kwargs.keys() else 1e-1,
        ent_coeff=kwargs['ent_coeff'] if 'ent_coeff' in kwargs.keys() else 0e-3,
        target_entropy=kwargs["target_entropy"] if 'target_entropy' in kwargs.keys() else -99.,
        entropy_lr=kwargs["entropy_lr"] if 'entropy_lr' in kwargs.keys() else 1e-4,
        initial_entropy_bonus=kwargs["initial_entropy_bonus"] if 'initial_entropy_bonus' in kwargs.keys() else 3e-3,
        entropy_lb=kwargs["entropy_lb"] if 'entropy_lb' in kwargs.keys() else -52,
        initial_entropy_lb=kwargs["initial_entropy_lb"] if 'initial_entropy_lb' in kwargs.keys() else -26,
        entropy_lb_ep=kwargs["entropy_lb_ep"] if 'entropy_lb_ep' in kwargs.keys() else 2000,
    )

    name = (f"ePPO_unstructured_qdiv50_tdiv5_fixedepsppo_"
            f"gamma099_hor150_"
            f"lr{agent_params['mu_lr']}_valuelr{agent_params['value_lr']}_bs{batch_size}_"
            f"constrlr{agent_params['constraint_lr']}_nep{n_episodes}_neppf{n_episodes_per_fit}_"
            f"neppol{agent_params['n_epochs_policy']}_epsppo{agent_params['eps_ppo']}_"
            f"sigmainit{agent_params['sigma_init_q']}q_{agent_params['sigma_init_t']}t_entlb{agent_params['entropy_lb']}_"
            f"entlbinit{agent_params['initial_entropy_lb']}_entlbep{agent_params['entropy_lb_ep']}_"
            f"nqcps{agent_params['n_q_cps']}_ntcps{agent_params['n_t_cps']}_seed{seed}")

    results_dir = os.path.join(results_dir, name)

    logger = Logger(log_name=env, results_dir=results_dir, seed=seed)

    if use_cuda:
        TorchUtils.set_default_device('cuda')

    #wandb_run = wandb.init(project="air_hockey_challenge_fullrange_still", config={}, dir=results_dir, name=name,
    #          group=f'{env}_{alg}_ePPO_verynewreward_dqddqscaling_undiscounted', tags=[str(env)])

    wandb_run = wandb.init(project="air_hockey_challenge_single", config={}, dir=results_dir, name=name,
              group=f'{env}_{alg}_ePPO_unstructured_discounted099', tags=[str(env)])

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
        gamma=0.99,
    )

    env, env_info_ = env_builder(env, n_envs, env_params)

    agent = agent_builder(env_info_, agent_params)
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhit_alphaclipm7_qddotdmul5qddiv30_hor125_gamma099_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-394.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhit_nodistloss_qddiv30qddotmul5gamma099_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.05_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed123/agent-123-444.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhit_flexibleqdotdfixed_betaqdot1em4qddot1em5_qddiv30qddotdmul5_gamma099_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-453.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhit_flexibleqdotdfixed_betaqdot1em4qddot1em5_qddiv30qddotdmul5_gamma099_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-193.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_TROhit_flexibleqdotdfixed_betaqdot1em4qddot1em5_qddiv30qddotdmul5_gamma099_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-158.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_gamma099_flexibleqdotd_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-542.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_flexibleqdotd_minalpham7_tighterdqddqviolationlimits_gamma099_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-248.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_gamma099_fixedentropyproj_verynewrewardnormv_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent_lb-99_nqcps11_ntcps10_seed444/agent-444-634.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_gamma099_fixedetropyproj_verynewrewardnormv_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.05_sigmainit0.1_ent_lb-77_nqcps11_ntcps10_seed444/agent-444-767.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_gamma099_fixedetropyproj_verynewrewardnormv_hor150_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.05_sigmainit0.1_ent_lb-77_nqcps11_ntcps10_seed444/agent-444-484.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_dqddqscaling_verynewrewardnormv_hor150_center_defaulttime07s_stillrandompose_initsigmaq01t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_nqcps11_ntcps10_seed444/agent-444-295.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_verynewrewardvnorm_smallerdqddqvioationbudget_puckcenter_defaulttime07s_stillrandompose_initsigmaq01t015nn_SACentropytarm99lr1em4init3em3_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_seed444/agent-444-339.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_verynewreward_smallerdqddqvioationbudget_puckcenter_defaulttime07s_stillrandompose_initsigmaq01t015nn_SACentropytarm99lr1em4init3em3_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_seed444/agent-444-765.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_take2_verynewreward_puckcenter_hor150_initsigmaq01t015_tarm99lr1em4init3em3_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_nqcps11_ntcps10_seed444/agent-444-238.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_verynewreward_center_defaulttime07s_horizon150_stillrandompose_initsigmaq003t015_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_nqcps11_ntcps10_seed444/agent-444-370.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_verynewreward_puckcenter_defaulttime07s_stillrandompose_initsigmaq003t015nn_SACentropytarm99lr1em4init3em3_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_seed444/agent-444-425.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_puckcenter_newreward_differentscaling10_biased_hor150_onlydistnogerhashitonlyclean_initsigmaddq1nn_tarm99lr1em4init3em3_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.001_nqcps11_ntcps10_seed444/agent-444-750.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillrandompose_initsigma02nn_nobigsigma_nn_SACentropytest_tarm33lr1em3init2em3_lr5e-05_valuelr0.0005_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.02_sigmainit0.1_ent0.0_seed444/agent-444-2745.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillrandompose_initsigma02nn_nobigsigma_nn_lr5e-05_valuelr0.0005_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.02_sigmainit0.1_ent0.001_seed444/7dof-hit/agent-444-681.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "logs/444/ePPO_stillrandompose_nn_lr5e-05_bs32_constrlr0.01_nep32_neppf32_neppol32_epsppo0.05_sigmainit0.1_ent0.0_seed444/7dof-hit/agent-444.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillrandompose10cm_nn_initsigma01nn_nobigsigma_lr5e-05_bs64_constrlr0.01_nep64_neppf64_neppol32_epsppo0.02_sigmainit0.1_ent0.0_seed444/7dof-hit/agent-444.msh")
    #agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillsamepose_nonn_sigma_lr0.01_bs32_constrlr0.01_nep32_neppf32_neppol4_epsppo0.02_sigmainit0.1_ent0.0_seed444_betapos1em5_betavel1em3_betaacc_1em4/7dof-hit/agent-444.msh")
    ##agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillsamepose_nonn_sigma_lr0.03_bs32_constrlr0.01_nep32_neppf32_neppol4_epsppo0.05_sigmainit0.1_ent0.0_seed444/7dof-hit/agent-444.msh")
    ##agent_path = os.path.join(os.path.dirname(__file__), "trained_models/ePPO_stillsamepose_nonn_lr0.03_bs32_constrlr0.01_nep32_neppf32_neppol4_epsppo0.02_sigmainit0.1_ent0.0_seed444/7dof-hit/agent-444.msh")

    #print("Load agent from: ", agent_path)
    #agent_ = agent_builder(env_info_, agent_params)
    #agent = Agent.load(agent_path)
    #agent.bsmp_agent.load_robot()
    ##agent.bsmp_agent._optimizer = torch.optim.Adam(agent.bsmp_agent.distribution.parameters(), lr=agent_params["mu_lr"])
    ##agent.bsmp_agent._epoch_no = 0
    ###agent.bsmp_agent.policy.env_info = env_info_
    ##agent.bsmp_agent.policy.optimizer = TrajectoryOptimizer(env_info_)
    #agent.bsmp_agent.policy.load_policy(env_info_)
    ##agent.bsmp_agent.policy.desired_ee_z = env_info_["robot"]["ee_desired_height"]
    ##agent.bsmp_agent.policy.joint_vel_limit = env_info_["robot"]["joint_vel_limit"][1] 
    ##agent.bsmp_agent.policy.joint_acc_limit = env_info_["robot"]["joint_acc_limit"][1] 
    ##agent.info.is_stateful = agent_.info.is_stateful
    ##agent.info.policy_state_shape = agent_.info.policy_state_shape


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
            print("Rs train: ", dataset_callback.get().undiscounted_return)
            print("Js train: ", dataset_callback.get().discounted_return)
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
            mean_duration = 0.

        # Evaluate
        J_det, R, success, c_avg, c_max, states, actions, time_to_hit, max_puck_vel = compute_metrics(core, eval_params)
        #assert False
        #wandb_plotting(core, states, actions, epoch)

        #entropy_lb = np.maximum(agent_params["initial_entropy_lb"] +
        #    (agent_params["entropy_lb"] - agent_params["initial_entropy_lb"]) * epoch / agent_params["entropy_lb_ep"], agent_params["entropy_lb"])
        #core.agent.bsmp_agent.distribution.set_e_lb(entropy_lb)

        if "logger_callback" in kwargs.keys():
            kwargs["logger_callback"](J_det, J_sto, V_sto, R, E, success, c_avg, c_max)

        # Write logging
        logger.log_numpy(J_det=J_det, J_sto=J_sto, V_sto=V_sto, VJ_bias=VJ_bias, R=R, E=E,
                         success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                         c_max=np.max(np.concatenate(list(c_max.values()))))
        logger.epoch_info(epoch, J_det=J_det, V_sto=V_sto, VJ_bias=VJ_bias, R=R, E=E,
                          success=success, c_avg=np.mean(np.concatenate(list(c_avg.values()))),
                          c_max=np.max(np.concatenate(list(c_max.values()))),
                          time_to_hit=time_to_hit, max_puck_vel=max_puck_vel)
        wandb.log({
            "Reward/": {"J_det": J_det, "J_sto": J_sto, "V_sto": V_sto, "VJ_bias": VJ_bias, "R": R, "success": success},
            #"Entropy/": {"E": E, "entropy_bonus": core.agent.bsmp_agent._log_entropy_bonus.exp().detach().numpy()},
            "Entropy/": {"E": E},
            "Constraints_sto/": {"avg/": {str(i): a for i, a in enumerate(constraints_violation_sto_mean)},
                                 "max/": {str(i): a for i, a in enumerate(constraints_violation_sto_max)}},
            "Constraints_det/": {"avg/": {str(i): a for i, a in enumerate(constraints_violation_det_mean)},
                                 "max/": {str(i): a for i, a in enumerate(constraints_violation_det_max)}},
            "Stats/": {"mean_duration": mean_duration, "time_to_hit": time_to_hit, "max_puck_vel": max_puck_vel},
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

        #if best_J_sto <= J_sto:
        #    best_J_sto = J_sto
        #    logger.log_agent(agent, epoch=epoch)

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
    #if "hit" in env_name:
    #    env_params["custom_reward_function"] = HitReward()

    #if "defend" in env_name:
    #    env_params["custom_reward_function"] = DefendReward()

    #if "prepare" in env_name:
    #    env_params["custom_reward_function"] = PrepareReward()

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

    if alg == "bsmp_eppo":
        bsmp_agent = build_agent_BSMPePPO(env_info, **agent_params)
    elif alg == "bsmp_eppo_return":
        bsmp_agent = build_agent_BSMPePPO_return(env_info, **agent_params)
    elif alg == "bsmp_eppo_unstructured":
        bsmp_agent = build_agent_BSMPePPO(env_info, **agent_params)
    elif alg == "promp":
        bsmp_agent = build_agent_ProMPePPO(env_info, **agent_params)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    return BSMPAgent(env_info, bsmp_agent)


def build_agent_BSMPePPO_return(env_info, **agent_params):

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
    n_trainable_q_stop_pts = n_q_pts - 6
    n_trainable_t_pts = n_t_pts
    n_trainable_t_stop_pts = n_t_pts

    mdp_info = env_info['rl_info']

    sigma_q_hit = 0.1 * torch.ones((n_trainable_q_pts, n_dim))
    sigma_q_stop = 0.1 * torch.ones((n_trainable_q_stop_pts, n_dim))
    sigma_t_hit = 0.15 * torch.ones((n_trainable_t_pts))
    sigma_t_stop = 0.15 * torch.ones((n_trainable_t_stop_pts))
    sigma_xy_stop = 0.10 * torch.ones((2))
    sigma = torch.cat([sigma_q_hit.reshape(-1), sigma_q_stop.reshape(-1), sigma_t_hit, sigma_t_stop, sigma_xy_stop]).type(torch.FloatTensor)

    mu_approximator = Regressor(TorchApproximator,
                                network=ConfigurationTimeNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts + n_dim * n_trainable_q_stop_pts + 2, n_trainable_t_pts + n_trainable_t_stop_pts))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=LogSigmaNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_dim * n_trainable_q_pts + n_trainable_t_pts + n_dim * n_trainable_q_stop_pts + 2 + n_trainable_t_stop_pts,))

    value_function_approximator = ValueNetwork(mdp_info.observation_space)

    policy = BSMPStoppingPolicy(env_info, env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end, robot_constraints)
    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])

    optimizer = {'class': optim.Adam,
                 'params': {'lr': agent_params["mu_lr"],
                            'weight_decay': 0.0}}

    value_function_optimizer = optim.Adam(value_function_approximator.parameters(), lr=agent_params["value_lr"])

    context_builder = None
    if dist.is_contextual:
        context_builder = IdentityContextBuilder()

    eppo_params = dict(n_epochs_policy=agent_params["n_epochs_policy"],
                       batch_size=agent_params["batch_size"],
                       eps_ppo=agent_params["eps_ppo"],
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

    sigma_q = agent_params["sigma_init_q"] * torch.ones((n_trainable_q_pts, n_dim))
    sigma_t = agent_params["sigma_init_t"] * torch.ones((n_trainable_t_pts))
    sigma = torch.cat([sigma_q.reshape(-1), sigma_t]).type(torch.FloatTensor)

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
                                output_shape=(n_dim * n_trainable_q_pts + n_trainable_t_pts,))

    value_function_approximator = ValueNetwork(mdp_info.observation_space)

    mu = torch.zeros(n_trainable_pts)
    if agent_params["alg"] == "bsmp_eppo_unstructured":
        policy = BSMPUnstructuredPolicy(env_info, env_info["dt"], n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin, n_pts_fixed_end, robot_constraints)
    else:
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
    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])
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
                       eps_ppo=agent_params["eps_ppo"],
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


def build_agent_ProMPePPO(env_info, **agent_params):

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
    n_dim = agent_params["n_dim"]
    n_trainable_q_pts = n_q_pts - 1
    n_trainable_pts = n_dim * n_trainable_q_pts

    mdp_info = env_info['rl_info']

    sigma = agent_params["sigma_init_q"] * torch.ones(n_trainable_pts)

    mu_approximator = Regressor(TorchApproximator,
                                network=ConfigurationNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_trainable_pts,))
    log_sigma_approximator = Regressor(TorchApproximator,
                                network=LogSigmaNetworkWrapper,
                                batch_size=1,
                                params={
                                        "input_space": mdp_info.observation_space,
                                        "init_sigma": sigma,
                                        },
                                input_shape=(mdp_info.observation_space.shape[0],),
                                output_shape=(n_trainable_pts,))

    value_function_approximator = ValueNetwork(mdp_info.observation_space)

    policy = ProMPPolicy(env_info, n_q_pts, n_dim)

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
    dist = DiagonalGaussianBSMPSigmaDistribution(mu_approximator, log_sigma_approximator, agent_params["entropy_lb"])
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
                       eps_ppo=agent_params["eps_ppo"],
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
    print("Rs val:", dataset.undiscounted_return)
    print("Js val:", dataset.discounted_return)

    eps_length = dataset.episodes_length
    success = 0
    current_idx = 0
    time_to_hit = []
    max_puck_vel = []
    for episode_len in eps_length:
        success += dataset.info["success"][current_idx + episode_len - 1]
        hit_time = dataset.info["hit_time"][current_idx + episode_len - 1]
        if hit_time is not None:
            time_to_hit.append(hit_time)
        max_puck_vel.append(np.max(dataset.info["puck_velocity"][current_idx:current_idx + episode_len]))
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
    action = dataset.action

    #state = state.reshape(dataset.n_episodes, int(state.shape[0] / dataset.n_episodes), state.shape[1])
    #action = action.reshape(dataset.n_episodes, int(action.shape[0] / dataset.n_episodes), 2, 7).transpose(0, 1, 3, 2)
    #puck, puck_dot, q, _, q_dot, _, q_ddot, _, _ = unpack_data_airhockey(state)
    #ee_pos, ee_rot = core.agent.bsmp_agent.compute_forward_kinematics(torch.tensor(q), torch.tensor(q_dot))
    #ee_pos = ee_pos.detach().numpy()#[0]
    #ee_rot = ee_rot.detach().numpy()#[0]
    #q_d = action[..., 0]
    #q_dot_d = action[..., 1]
    #ee_pos_d, ee_rot_d = core.agent.bsmp_agent.compute_forward_kinematics(torch.tensor(q_d), torch.tensor(q_dot_d))
    #ee_pos_d = ee_pos_d.detach().numpy()#[0]
    #ee_rot_d = ee_rot_d.detach().numpy()#[0]

    #q_dot_l = core.agent.bsmp_agent.robot_constraints['q_dot']
    #q_ddot_l = core.agent.bsmp_agent.robot_constraints['q_ddot']
    #q_dot_ = torch.tensor(q_dot)
    #q_ddot_ = torch.tensor(q_ddot)
    #ee_pos_ = torch.tensor(ee_pos)
    #q_dot_l_ = torch.tensor(q_dot_l)[None]
    #q_ddot_l_ = torch.tensor(q_ddot_l)[None]
    #dt = core.agent.bsmp_agent.policy.dt * torch.ones(q_dot_.shape[0], 1, 1)
    #q_dot_loss = limit_loss(torch.abs(q_dot_), dt, q_dot_l_)
    #q_ddot_loss = limit_loss(torch.abs(q_ddot_), dt, q_ddot_l_)

    #x_ee_loss_low = limit_loss(core.agent.bsmp_agent.robot_constraints["x_ee_lb"], dt, ee_pos_[..., 0])
    #y_ee_loss_low = limit_loss(core.agent.bsmp_agent.robot_constraints["y_ee_lb"], dt, ee_pos_[..., 1])
    #y_ee_loss_high = limit_loss(ee_pos_[..., 1], dt, core.agent.bsmp_agent.robot_constraints["y_ee_ub"])
    #z_ee_loss = equality_loss(ee_pos_[..., 2], dt, core.agent.bsmp_agent.robot_constraints["z_ee"])
    #idx = 0
    #plt.subplot(211)
    #plt.plot(ee_pos[idx, :, 0], ee_pos[idx, :, 1], label="actual")
    #plt.plot(ee_pos_d[idx, :, 0], ee_pos_d[idx, :, 1], label="desired")
    #plt.subplot(212)
    #plt.plot(ee_pos[idx, :, 2], label="actual")
    #plt.plot(ee_pos_d[idx, :, 2], label="desired")
    #plt.legend()
    #plt.show()
    #for i in range(q.shape[-1]):
    #    plt.subplot(3, 7, 1+i)
    #    plt.plot(q[idx, :, i], label="actual")
    #    plt.plot(q_d[idx, :, i], label="desired")
    #for i in range(q.shape[-1]):
    #    plt.subplot(3, 7, 1+i+7)
    #    plt.plot(q_dot[idx, :, i], label="actual")
    #    plt.plot(q_dot_d[idx, :, i], label="desired")
    #    plt.plot(q_dot_l[i] * np.ones_like(q_dot[idx, :, i]), label="limit")
    #    plt.plot(-q_dot_l[i] * np.ones_like(q_dot[idx, :, i]), label="limit")
    #for i in range(q.shape[-1]):
    #    plt.subplot(3, 7, 1+i+14)
    #    plt.plot(q_ddot[idx, :, i], label="actual")
    #    plt.plot(q_ddot_l[i] * np.ones_like(q_ddot[idx, :, i]), label="limit")
    #    plt.plot(-q_ddot_l[i] * np.ones_like(q_ddot[idx, :, i]), label="limit")
    #plt.legend()
    #plt.show()

    return J, R, success, c_avg, c_max, state, action, np.mean(time_to_hit), np.mean(max_puck_vel)


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
