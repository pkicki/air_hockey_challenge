import numpy as np
import torch
from examples.rl.bsmp.bspline import BSpline
from scipy.interpolate import interp1d

from mushroom_rl.policy import Policy

from examples.rl.bsmp.utils import unpack_data_airhockey


class BSMPPolicy(Policy):
    def __init__(self, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin=1, n_pts_fixed_end=1):
        self.dt = dt
        self.n_dim = n_dim
        self._n_q_pts = n_q_pts
        self._n_t_pts = n_t_pts
        self._n_pts_fixed_begin = n_pts_fixed_begin
        self._n_pts_fixed_end = n_pts_fixed_end
        self._n_trainable_q_pts = self._n_q_pts - (self._n_pts_fixed_begin + self._n_pts_fixed_end)
        self._n_trainable_t_pts = self._n_t_pts

        self._q_bsp = BSpline(self._n_q_pts)
        self._t_bsp = BSpline(self._n_t_pts)
        self._qdd1 = self._q_bsp.ddN[0, 0, 0]
        self._qdd2 = self._q_bsp.ddN[0, 0, 1]
        self._qdd3 = self._q_bsp.ddN[0, 0, 2]
        self._qd1 = self._q_bsp.dN[0, 0, 1]
        self._td1 = self._t_bsp.dN[0, 0, 1]

        self.q = None
        self.q_dot = None
        self.q_ddot = None
        self.duration = None

        self._weights = None
        self._trainable_q_cps = None
        self._trainable_t_cps = None

        policy_state_shape = tuple()
        super().__init__(policy_state_shape)

        self._add_save_attr(
            _dt='primitive',
        )

    def unpack_context(self, context):
        puck, puck_dot, q0, qd, dq0, dqd, ddq0, ddqd, opponent_mallet = unpack_data_airhockey(torch.tensor(context))
        return q0[:, None], qd[:, None], dq0[:, None], dqd[:, None], ddq0[:, None], ddqd[:, None]

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = self.unpack_context(torch.tensor(context))
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [trainable_q_cps] + q_end[:self._n_pts_fixed_end][::-1], axis=-2)
        #q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.detach().numpy(), trainable_t_cps.detach().numpy())
        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)
        return q, q_dot, q_ddot, t, dt, duration


    def reset(self, initial_state=None):
        if initial_state is None:
            return None
        else:
            if len(initial_state.shape) == 1:
                initial_state = initial_state[None]
            q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory_from_theta(self._weights, initial_state)
            self.q = interp1d(t[0], q[0], axis=0)
            self.q_dot = interp1d(t[0], q_dot[0], axis=0)
            self.q_ddot = interp1d(t[0], q_ddot[0], axis=0)
            self.duration = duration[0]
            return torch.tensor([0], dtype=torch.int32)
        

    def draw_action(self, state, policy_state=None):
        """
        Args:
            state (ndarray): state of the system
            policy_state (ndarray, None): the policy internal state.

        Returns:
            numpy.ndarray, (3, num_joints): The desired [Positions, Velocities, Acceleration] of the
            next step. The environment will take first two arguments of the to control the robot.
            The third array is used for the training of the SAC as the output is acceleration. This
            action tuple will be saved in the dataset buffer
        """
        assert policy_state is not None
        t = min(policy_state[0] * self.dt, self.duration)
        q = self.q(t)
        q_dot = self.q_dot(t)
        q_ddot = self.q_ddot(t)
        policy_state[0] += 1
        action = np.stack([q, q_dot, q_ddot], axis=-2) 
        action = torch.tensor(action, dtype=torch.float32)
        return action, torch.tensor(policy_state)

    def extract_qt(self, x):
        # TODO: make it suitable for parallel envs
        q_cps = x[:, :self._n_trainable_q_pts * self.n_dim]
        t_cps = x[:, self._n_trainable_q_pts * self.n_dim:]
        q_cps = q_cps.reshape(-1, self._n_trainable_q_pts, self.n_dim)
        t_cps = t_cps.reshape(-1, self._n_trainable_t_pts, 1)
        return q_cps, t_cps

    def set_weights(self, weights):
        self._weights = weights

    def compute_boundary_control_points(self, dtau_dt, q0, q_dot_0, q_ddot_0, qd, q_dot_d, q_ddot_d):
        q1 = q_dot_0 / dtau_dt[:, :1] / self._qd1 + q0
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self._qd1
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self._qd1 * self._td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, None]) / dtau_dt[:, :1]
              - self._qdd1 * q0 - self._qdd2 * q1) / self._qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
                self._qd1 * self._td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, None]) / dtau_dt[:, -1:]
               - self._qdd1 * qd - self._qdd2 * qm1) / self._qdd3
        return q1, q2, qm2, qm1

    def compute_boundary_control_points_exp(self, dtau_dt, q0, q_dot_0, q_ddot_0, qd, q_dot_d, q_ddot_d):
        q1 = q_dot_0 / (torch.exp(dtau_dt[:, :1]) * self._qd1) + q0
        qm1 = qd - q_dot_d / (torch.exp(dtau_dt[:, -1:]) * self._qd1)
        q2 = (q_ddot_0 / torch.exp(dtau_dt[:, :1])**2
              - self._qd1 * self._td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, None]
              - self._qdd1 * q0
              - self._qdd2 * q1) / self._qdd3
        qm2 = (q_ddot_d / torch.exp(dtau_dt[:, -1:])**2
               - self._qd1 * self._td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, None]
               - self._qdd1 * qd
               - self._qdd2 * qm1) / self._qdd3
        return q1, q2, qm2, qm1


    def compute_trajectory(self, q_cps, t_cps, differentiable=False):
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

        #dtau_dt = tN @ t_cps
        #ddtau_dtt = tdN @ t_cps
        dtau_dt = torch.exp(tN @ t_cps) if differentiable else np.exp(tN @ t_cps)
        ddtau_dtt = dtau_dt * (tdN @ t_cps)

        # ensure that dt is non-negative
        #dt = 1. / np.abs(dtau_dt[..., 0]) / dtau_dt.shape[-2] if not differentiable else 1. / torch.abs(dtau_dt[..., 0]) / dtau_dt.shape[-2]
        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[-2]
        t = np.cumsum(dt, axis=-1) - dt[..., :1] if not differentiable else torch.cumsum(dt, dim=-1) - dt[..., :1]
        duration = t[:, -1]

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
        return q, q_dot, q_ddot, t, dt, duration
