from copy import copy
import os
from time import perf_counter

from mushroom_rl.core import Agent
from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.rl_utils.optimizers import AdaptiveOptimizer
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy.gaussian_policy import DiagonalGaussianPolicy
from mushroom_rl.utils.torch import TorchUtils

import torch
import numpy as np
from scipy.interpolate import interp1d
from bsmp.bspline import BSpline
from bsmp.bspline_approximator import BSplineApproximatorAirHockey, BSplineApproximatorAirHockeyWrapper
from bsmp.utils import equality_loss, limit_loss

from differentiable_robot_model.robot_model import DifferentiableKUKAiiwa, DifferentiableRobotModel

from examples.rl.bsmp.bsmp_policy import BSMPPolicy
from examples.rl.bsmp.bspline_timeoptimal_approximator import BSplineFastApproximatorAirHockeyWrapper


class GaussianDiagonalDistributionVectorized(GaussianDiagonalDistribution):
    def __init__(self, mu, std):
        self._mu = mu
        self._std = std

        self._add_save_attr(
            _mu='numpy',
            _std='numpy'
        )

    def sample(self):
        return np.stack([np.random.multivariate_normal(mu, np.diag(self._std**2)) for mu in self._mu], axis=0)


class BSMP(Agent):
    """
    """

    def __init__(self, mdp_info, robot_constraints, dt, n_q_cps, n_t_cps, n_pts_fixed_begin, n_pts_fixed_end,
                 n_dim, sigma_init, sigma_eps, mu_lr, constraint_lr, **kwargs):
        """
        Constructor.

        Args:
            policy (ParametricPolicy): the policy to use.

        """
        self._n_q_pts = n_q_cps
        self._n_t_pts = n_t_cps
        self._q_bsp = BSpline(self._n_q_pts)
        self._t_bsp = BSpline(self._n_t_pts)
        self._n_pts_fixed_begin = n_pts_fixed_begin
        self._n_pts_fixed_end = n_pts_fixed_end
        self._n_trainable_q_pts = self._n_q_pts - (self._n_pts_fixed_begin + self._n_pts_fixed_end)
        self._n_trainable_t_pts = self._n_t_pts
        self._n_dim = n_dim
        self._n_trainable_pts = self._n_dim * self._n_trainable_q_pts + self._n_trainable_t_pts


        self.robot_constraints = robot_constraints
        self.alphas = np.array([0.] * 18)
        self.violation_limits = np.array([1e-3] * 7 + [1e-4] * 7 + [1e-3] * 4)
        self.constraint_lr = constraint_lr
        self.constraint_losses = []
        self.constraint_losses_log = []

        self.mu_approximator = Regressor(TorchApproximator,
                         #network=BSplineApproximatorAirHockeyWrapper,
                         network=BSplineFastApproximatorAirHockeyWrapper,
                         batch_size=1,
                         params={"q_bsp": self._q_bsp,
                                 "t_bsp": self._t_bsp,
                                 "n_dim": self._n_dim,
                                 "n_pts_fixed_begin": self._n_pts_fixed_begin,
                                 "n_pts_fixed_end": self._n_pts_fixed_end,
                                 "input_space": mdp_info.observation_space,
                                 "q_dot_limit": robot_constraints["q_dot"],
                                 "q_ddot_limit": robot_constraints["q_ddot"],
                                 },
                         input_shape=(mdp_info.observation_space.shape[0],),
                         output_shape=(self._n_trainable_pts,))

        self.mu_optimizer = torch.optim.Adam(self.mu_approximator.model.network.parameters(), lr=mu_lr)

        self.q_log_t_cps_sigma_trainable = np.log(sigma_init * np.ones((self._n_trainable_pts,)))
        self.sigma_optimizer = AdaptiveOptimizer(eps=sigma_eps)

        self.urdf_path = os.path.join(os.path.dirname(__file__), "iiwa_striker.urdf")
        self.load_robot()


        policy = BSMPPolicy(dt=dt)

        super().__init__(mdp_info, policy)

        self._add_save_attr(q_log_t_cps_sigma_trainable='numpy',
                            _q_bsp='pickle',
                            _t_bsp='pickle',
                            _n_pts_fixed_begin='primitive',
                            _n_pts_fixed_end='primitive',
                            _n_q_pts='primitive',
                            _n_t_pts='primitive',
                            _n_trainable_q_pts='primitive',
                            _n_trainable_t_pts='primitive',
                            _n_dim='primitive',
                            _n_trainable_pts='primitive',
                            alphas='numpy',
                            violation_limits='numpy',
                            constraint_lr='primitive',
                            sigma_optimizer='torch',
                            mu_approximator='mushroom',
                            mu_optimizer='torch',
                            robot_constraints='pickle',
                            urdf_path='primitive',
                            constraint_losses='pickle',
                            constraint_losses_log='pickle',
                            )

    def load_robot(self):
        self.robot = DifferentiableRobotModel(urdf_path=self.urdf_path, name="iiwa", device="cpu")

    def _unpack_qt(self, qt, trainable=False):
        n_q_pts = self._n_trainable_q_pts if trainable else self._n_q_pts
        q = qt[..., :self._n_dim * n_q_pts]
        t = qt[..., self._n_dim * n_q_pts:]
        return q, t

    def _compute_trajectory(self, q_cps, t_cps, differentiable=False):
        qN = self._q_bsp.N
        qdN = self._q_bsp.dN
        qddN = self._q_bsp.ddN
        tN = self._t_bsp.N
        tdN = self._t_bsp.dN
        if differentiable:
            qN = torch.Tensor(qN)
            qdN = torch.Tensor(qdN)
            qddN = torch.Tensor(qddN)
            tN = torch.Tensor(tN)
            tdN = torch.Tensor(tdN)

        q = qN @ q_cps
        q_dot_tau = qdN @ q_cps
        q_ddot_tau = qddN @ q_cps

        dtau_dt = tN @ t_cps
        ddtau_dtt = tdN @ t_cps

        # todo check if the indexing is general
        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[-2]
        t = np.cumsum(dt, axis=-1) - dt[..., :1] if not differentiable else torch.cumsum(dt, dim=-1) - dt[..., :1]
        duration = t[:, -1]

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
        return q, q_dot, q_ddot, t, dt, duration

    def update_alphas(self):
        constraint_losses = torch.mean(torch.cat(self.constraint_losses, dim=0), dim=0)
        alphas_update = self.constraint_lr * np.log(
            (constraint_losses.detach().numpy() + self.violation_limits * 1e-1) / self.violation_limits)
        self.alphas += alphas_update
        self.constraint_losses_log = constraint_losses
        self.constraint_losses = []

    def compute_mean_trajectory(self, state):
        q_cps_mu, log_t_cps_mu = self.query_mu_approximator(state)
        q_cps_mu = q_cps_mu[0].cpu().detach().numpy()
        log_t_cps_mu = log_t_cps_mu[0, ..., None].cpu().detach().numpy()
        q_cps = q_cps_mu
        t_cps = np.exp(log_t_cps_mu)
        return self._compute_trajectory(q_cps, t_cps)

    def query_mu_approximator(self, state):
        state = state.astype(np.float32)
        q_cps_mu, t_cps_mu = self.mu_approximator.model.network(torch.from_numpy(state).to(TorchUtils.get_device()))
        return q_cps_mu, t_cps_mu

    def compute_trajectory(self, state):
        q_cps_mu, log_t_cps_mu = self.query_mu_approximator(state)

        q_cps_mu_trainable = q_cps_mu[:, self._n_pts_fixed_begin:]
        if self._n_pts_fixed_end:
            q_cps_mu_trainable = q_cps_mu[self._n_pts_fixed_begin:-self._n_pts_fixed_end]
        q_cps_mu_trainable = q_cps_mu_trainable.reshape((-1, self._n_trainable_q_pts * self._n_dim))

        log_t_cps_mu_trainable = log_t_cps_mu.reshape((-1, self._n_trainable_t_pts))
        q_log_t_cps_mu_trainable = torch.cat([q_cps_mu_trainable, log_t_cps_mu_trainable], axis=-1)
        q_log_t_cps_mu = torch.cat([q_cps_mu.reshape((-1, self._n_q_pts * self._n_dim)),
                                    log_t_cps_mu.reshape((-1, self._n_t_pts))], axis=-1)

        q_log_t_cps_dist = GaussianDiagonalDistributionVectorized(q_log_t_cps_mu_trainable.detach().numpy(),
                                                                  np.exp(self.q_log_t_cps_sigma_trainable))

        q_log_t_cps_trainable = q_log_t_cps_dist.sample()
        q_cps_trainable, t_log_cps_trainable = self._unpack_qt(q_log_t_cps_trainable, trainable=True)

        q_cps = q_cps_trainable.reshape((-1, self._n_trainable_q_pts , self._n_dim))
        q_cps_mu = q_cps_mu.detach().cpu().numpy()
        if self._n_pts_fixed_end:
            q_cps = np.concatenate([q_cps_mu[:, :self._n_pts_fixed_begin], q_cps, q_cps_mu[:, -self._n_pts_fixed_end:]], axis=-2)
        else:
            q_cps = np.concatenate([q_cps_mu[:, :self._n_pts_fixed_begin], q_cps], axis=-2)
        t_log_cps = t_log_cps_trainable.reshape((-1, self._n_t_pts, 1))
        t_cps = np.exp(t_log_cps)

        q, q_dot, q_ddot, t, dt, duration = self._compute_trajectory(q_cps, t_cps)

        action_q = [interp1d(t[i], q[i], axis=0) for i in range(state.shape[0])]
        action_q_dot = [interp1d(t[i], q_dot[i], axis=0) for i in range(state.shape[0])]
        action_q_ddot = [interp1d(t[i], q_ddot[i], axis=0) for i in range(state.shape[0])]
        trajectory = [dict(q=action_q[i], q_dot=action_q_dot[i], q_ddot=action_q_ddot[i], duration=duration[i], dt=dt[i])
                      for i in range(state.shape[0])]
        theta = [dict(theta=q_log_t_cps_trainable[i], mu_trainable=q_log_t_cps_mu_trainable[i], mu=q_log_t_cps_mu[i])
                 for i in range(state.shape[0])]
        return trajectory, theta

    def fit(self, dataset, **info):
        theta = np.array([theta["theta"] for theta in dataset.theta_list])
        q_log_t_cps_mu_trainable = torch.stack([theta["mu_trainable"] for theta in dataset.theta_list])
        q_log_t_cps_mu = torch.stack([theta["mu"] for theta in dataset.theta_list])
        Jep = np.array(dataset.discounted_return)

        print(Jep.shape, theta.shape, q_log_t_cps_mu_trainable.shape, q_log_t_cps_mu.shape)

        self._update(Jep, theta, q_log_t_cps_mu, q_log_t_cps_mu_trainable)

    def _update(self, Jep, theta, mu, mu_trainable):
        baseline_num_list = list()
        baseline_den_list = list()
        diff_log_dist_list = list()

        # Compute derivatives of distribution and baseline components
        for i in range(len(Jep)):
            J_i = Jep[i]
            theta_i = theta[i]
            distribution_i = GaussianDiagonalDistribution(mu_trainable[i].detach().numpy(), np.exp(self.q_log_t_cps_sigma_trainable))

            diff_log_dist = distribution_i.diff_log(theta_i)
            diff_log_dist2 = diff_log_dist ** 2

            diff_log_dist_list.append(diff_log_dist)
            baseline_num_list.append(J_i * diff_log_dist2)
            baseline_den_list.append(diff_log_dist2)

        # Compute baseline
        baseline = np.mean(baseline_num_list, axis=0) / \
                   np.mean(baseline_den_list, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.

        # Prepare the constrint limits tensors
        q_dot_limits = torch.Tensor(self.robot_constraints['q_dot'])[None, None]
        q_ddot_limits = torch.Tensor(self.robot_constraints['q_ddot'])[None, None]

        # All constraint losses computation organized in a single function
        def compute_constraint_losses(x):
            q_cps, t_log_cps = self._unpack_qt(x)
            q_cps = q_cps.reshape((x.shape[0], self._n_q_pts, self._n_dim))
            t_log_cps = t_log_cps.reshape((x.shape[0], self._n_t_pts, 1))
            t_cps = torch.exp(t_log_cps)
            q, q_dot, q_ddot, t, dt, duration = self._compute_trajectory(q_cps, t_cps, differentiable=True)

            dt_ = dt[..., None]
            q_dot_loss = limit_loss(torch.abs(q_dot), dt_, q_dot_limits)
            q_ddot_loss = limit_loss(torch.abs(q_ddot), dt_, q_ddot_limits)

            q_ = q.reshape((-1, self._n_dim))
            q_ = torch.cat([q_, torch.zeros((q_.shape[0], 9 - q_.shape[1]))], dim=-1)
            ee_pos, ee_quat = self.robot.compute_forward_kinematics(q_, "F_striker_tip")
            ee_pos = ee_pos.reshape((q.shape[0], q.shape[1], 3))
            ee_quat = ee_quat.reshape((q.shape[0], q.shape[1], 4))

            x_ee_loss_low = limit_loss(self.robot_constraints["x_ee_lb"], dt, ee_pos[..., 0])[..., None]
            y_ee_loss_low = limit_loss(self.robot_constraints["y_ee_lb"], dt, ee_pos[..., 1])[..., None]
            y_ee_loss_high = limit_loss(ee_pos[..., 1], dt, self.robot_constraints["y_ee_ub"])[..., None]
            z_ee_loss = equality_loss(ee_pos[..., 2], dt, self.robot_constraints["z_ee"])[..., None]

            constraint_losses = torch.cat([q_dot_loss, q_ddot_loss, x_ee_loss_low, y_ee_loss_low,
                                           y_ee_loss_high, z_ee_loss], dim=-1)
            return constraint_losses

        #constraint_losses_mu = compute_constraint_losses(mu)
        #std = torch.Tensor(self.compute_full_std())[None]
        #constraint_losses_mu_pstd = compute_constraint_losses(mu + std)
        #constraint_losses_mu_mstd = compute_constraint_losses(mu - std)
        #constraint_losses = constraint_losses_mu + constraint_losses_mu_pstd + constraint_losses_mu_mstd

        # Constraint losses computed from the means returned by the policy
        constraint_losses = compute_constraint_losses(mu)
        self.constraint_losses.append(constraint_losses)
        constraint_loss = torch.exp(torch.Tensor(self.alphas))[None] * constraint_losses
        constraint_loss = torch.sum(constraint_loss, dim=-1)

        # REINFORCE-like loss computed from samples
        task_reward = ((torch.from_numpy(theta) - mu_trainable) / torch.from_numpy(
            np.exp(self.q_log_t_cps_sigma_trainable))) ** 2 \
                      * torch.from_numpy(Jep - np.mean(Jep, axis=0, keepdims=True))[..., None]
        task_loss = torch.sum(task_reward, dim=-1)
        loss = task_loss + constraint_loss
        mean_loss = torch.mean(loss)
        self.mu_optimizer.zero_grad()
        mean_loss.backward()
        self.mu_optimizer.step()

        # PGPE update of the sigmas
        grad_J_list = list()
        for i in range(len(Jep)):
            diff_log_dist = diff_log_dist_list[i]
            J_i = Jep[i]

            grad_J_list.append(diff_log_dist * (J_i - baseline))

        grad_J = np.stack(grad_J_list)

        grad_J_sigma = grad_J[:, self._n_trainable_pts:]
        self.q_log_t_cps_sigma_trainable = self.sigma_optimizer(self.q_log_t_cps_sigma_trainable,
                                                                np.mean(grad_J_sigma, axis=0) * np.exp(
                                                                    self.q_log_t_cps_sigma_trainable))
