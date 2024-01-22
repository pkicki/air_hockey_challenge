import os
import numpy as np
import torch
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from examples.rl.bsmp.bspline import BSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from mushroom_rl.policy import Policy

from examples.rl.bsmp.utils import unpack_data_airhockey


class BSMPPolicy(Policy):
    def __init__(self, env_info, dt, n_q_pts, n_dim, n_t_pts, n_pts_fixed_begin=1, n_pts_fixed_end=1, robot_constraints=None):
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

        self._traj_no = 0
        self._robot_constraints = robot_constraints

        self.env_info = env_info
        self.optimizer = TrajectoryOptimizer(self.env_info)

        policy_state_shape = (1,)
        super().__init__(policy_state_shape)

        self._add_save_attr(
            dt='primitive',
            n_dim='primitive',
            _n_q_pts='primitive',
            _n_t_pts='primitive',
            _n_pts_fixed_begin='primitive',
            _n_pts_fixed_end='primitive',
            _n_trainable_q_pts='primitive',
            _n_trainable_t_pts='primitive',
            _q_bsp='pickle',
            _t_bsp='pickle',
            _qdd1='primitive',
            _qdd2='primitive',
            _qdd3='primitive',
            _qd1='primitive',
            _td1='primitive',
            _traj_no='primitive'
            env_info='pickle',
            optimizer='pickle',
        )

    def unpack_context(self, context):
        if context is None:
            raise NotImplementedError
            puck = torch.tensor([[1.01, 0., 0.]])
            q_0 = torch.tensor([[0., -0.196067, 0., -1.84364, 0., 0.970422, 0.]])
            q_d = torch.zeros((1, self.n_dim))
            q_dot_0 = torch.zeros((1, self.n_dim))
            q_dot_d = torch.zeros((1, self.n_dim))
            q_ddot_0 = torch.zeros((1, self.n_dim))
            q_ddot_d = torch.zeros((1, self.n_dim))
        else:
            puck, puck_dot, q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, opponent_mallet = unpack_data_airhockey(torch.tensor(context))
        return q_0[:, None], q_d[:, None], q_dot_0[:, None], q_dot_d[:, None], q_ddot_0[:, None], q_ddot_d[:, None], puck

    #def compute_trajectory_from_theta(self, theta, context):
    #    q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d = self.unpack_context(context)
    #    trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
    #    q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
    #                                                                q_d, q_dot_d, q_ddot_d)
    #    q_begin = [q_0, q1, q2]
    #    q_end = [q_d, qm1, qm2]
    #    q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_0 + torch.pi * trainable_q_cps] + q_end[:self._n_pts_fixed_end][::-1], axis=-2)
    #    #q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.detach().numpy(), trainable_t_cps.detach().numpy())
    #    #q_cps_ = q_cps.detach().numpy()[0]
    #    #t_cps_ = trainable_t_cps.detach().numpy()[0]
    #    #for i in range(self.n_dim):
    #    #    plt.subplot(1, 8, 1+i)
    #    #    plt.plot(q_cps_[:, i])
    #    #plt.subplot(1, 8, 1+self.n_dim)
    #    #plt.plot(t_cps_)
    #    ##plt.show()
    #    #plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"imgs/cps_{self._traj_no}.png"))
    #    #plt.clf()

    #    q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)
    #    #q_ = q.detach().numpy()[0]
    #    #q_dot_ = q_dot.detach().numpy()[0]
    #    #q_ddot_ = q_ddot.detach().numpy()[0]
    #    #t_ = t.detach().numpy()[0]
    #    #qdl = self._robot_constraints['q_dot']
    #    #qddl = self._robot_constraints['q_ddot']
    #    #for i in range(self.n_dim):
    #    #    plt.subplot(3, 7, 1+i)
    #    #    plt.plot(t_, q_[:, i])
    #    #    plt.subplot(3, 7, 1+i+self.n_dim)
    #    #    plt.plot(t_, q_dot_[:, i])
    #    #    plt.plot([t_[0], t_[-1]], [qdl[i], qdl[i]], 'r--')
    #    #    plt.plot([t_[0], t_[-1]], [-qdl[i], -qdl[i]], 'r--')
    #    #    plt.subplot(3, 7, 1+i+2*self.n_dim)
    #    #    plt.plot(t_, q_ddot_[:, i])
    #    #    plt.plot([t_[0], t_[-1]], [qddl[i], qddl[i]], 'r--')
    #    #    plt.plot([t_[0], t_[-1]], [-qddl[i], -qddl[i]], 'r--')
    #    #plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"imgs/traj_{self._traj_no}.png"))
    #    #plt.clf()
    #    self._traj_no += 1
    #    return q, q_dot, q_ddot, t, dt, duration

    def compute_trajectory_from_theta(self, theta, context):
        q_0, q_d, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, puck = self.unpack_context(context)
        trainable_q_cps, trainable_t_cps = self.extract_qt(theta)
        #trainable_q_cps = torch.tanh(trainable_q_cps/10.) * np.pi
        middle_trainable_q_pts = torch.tanh(trainable_q_cps[:, :-3]/10.) * np.pi
        trainable_q_d = torch.tanh(trainable_q_cps[:, -1:]/10.) * np.pi
        trainable_q_ddot_d = torch.tanh(trainable_q_cps[:, -3:-2]) * torch.tensor(self.env_info['robot']['joint_acc_limit'][1])
        trainable_delta_angle = torch.tanh(trainable_q_cps[:, -2:-1, -1]/10.) * np.pi/2.
        trainable_scale = torch.sigmoid(trainable_q_cps[:, -2, -2])[:, None, None]


        x_cur = forward_kinematics(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'], q_0[0, 0])[0]

        puck_pos = puck.detach().numpy()
        goal = np.array([2.484, 0., 0.])
        # Compute the vector that shoot the puck directly to the goal
        vec_puck_goal = (goal - puck_pos) / np.linalg.norm(goal - puck_pos)
        #x_des = puck_pos - (self.env_info['mallet']['radius'] + self.env_info['puck']['radius']) * vec_puck_goal
        x_des = puck_pos# - (self.env_info['mallet']['radius'] + self.env_info['puck']['radius']) * vec_puck_goal
        x_des[:, -1] = self.env_info['robot']['ee_desired_height'] - 0.03# - self.env_info['robot']['universal_height']
        r1 = torch.cat([torch.cos(trainable_delta_angle), -torch.sin(trainable_delta_angle), torch.zeros_like(trainable_delta_angle)], axis=-1)
        r2 = torch.cat([torch.sin(trainable_delta_angle), torch.cos(trainable_delta_angle), torch.zeros_like(trainable_delta_angle)], axis=-1)
        r3 = torch.cat([torch.zeros_like(trainable_delta_angle), torch.zeros_like(trainable_delta_angle), torch.ones_like(trainable_delta_angle)], axis=-1)
        R = torch.stack([r1, r2, r3], axis=-2)
        #R = torch.tensor([[torch.cos(trainable_delta_angle), -torch.sin(trainable_delta_angle), 0.],
        #                    [torch.sin(trainable_delta_angle), torch.cos(trainable_delta_angle), 0.],
        #                    [0., 0., 1.]])[None]
        v_des = (R @ torch.tensor(vec_puck_goal)[..., None])[..., 0]

        q_d_s = []
        for k in range(q_0.shape[0]):
            success, q_d = self.optimizer.solve_hit_config(x_des[k], v_des.detach().numpy()[k], q_0.detach().numpy()[k, 0])
            q_d_s.append(q_d)
        q_d_bias = torch.tensor(q_d_s)[:, None]
        q_d = trainable_q_d + q_d_bias
        q_dot_d_s = []
        for k in range(q_0.shape[0]):
            q_dot_d = (torch.linalg.pinv(torch.tensor(self.optimizer.jacobian(q_d.detach().numpy()[k, 0])))[:, :3] @ v_des.T)[..., 0]
            q_dot_d_s.append(q_dot_d)
        q_dot_d_bias = torch.stack(q_dot_d_s, dim=0)[:, None]
        scale = 1. / torch.max(torch.abs(q_dot_d_bias) / torch.tensor(self.env_info['robot']['joint_vel_limit'][1]), axis=-1, keepdim=True)[0]
        q_dot_d = q_dot_d_bias * scale #* trainable_scale
        #q_d = trainable_q_cps[:, -1:] + q_d_bias
        #q_dot_d = trainable_q_cps[:, -2:-1] + q_dot_d_bias

        # hax
        #q_d = trainable_q_cps[:, -1:] + q_0
        #q_dot_d = trainable_q_cps[:, -2:-1]
        #q_ddot_d = trainable_q_cps[:, -3:-2]
        #trainable_q_cps = trainable_q_cps[:, :-3]
        q_ddot_d = trainable_q_ddot_d
        q1, q2, qm2, qm1 = self.compute_boundary_control_points_exp(trainable_t_cps, q_0, q_dot_0, q_ddot_0,
                                                                    q_d, q_dot_d, q_ddot_d)
        q_begin = [q_0, q1, q2]
        q_end = [q_d, qm1, qm2]
        #q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_0 + torch.pi * trainable_q_cps] + q_end[::-1], axis=-2)

        s = torch.linspace(0., 1., middle_trainable_q_pts.shape[1]+6)[None, 3:-3, None]
        q_b = q_0 * (1 - s) + q_d * s
        q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + middle_trainable_q_pts] + q_end[::-1], axis=-2)

        #s = torch.linspace(0., 1., trainable_q_cps.shape[1])[None, :, None]
        #q_b = q_0 * (1 - s) + q_d * s
        #q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_b + trainable_q_cps] + q_end[::-1], axis=-2)

        #q_cps = torch.cat(q_begin[:self._n_pts_fixed_begin] + [q_0 + torch.pi * trainable_q_cps] + q_end[:self._n_pts_fixed_end][::-1], axis=-2)
        q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory(q_cps.to(torch.float32), trainable_t_cps.to(torch.float32), differentiable=True)
        self._traj_no += 1
        return q, q_dot, q_ddot, t, dt, duration


    def reset(self, initial_state=None):
        if initial_state is None:
            return None
        else:
            if len(initial_state.shape) == 1:
                initial_state = initial_state[None]
            q, q_dot, q_ddot, t, dt, duration = self.compute_trajectory_from_theta(self._weights, initial_state)
            q = q.detach().numpy()
            q_dot = q_dot.detach().numpy()
            q_ddot = q_ddot.detach().numpy()
            t = t.detach().numpy()
            duration = duration.detach().numpy()
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
        t = policy_state[0] * self.dt
        if t <= self.duration:
            q = self.q(t)
            q_dot = self.q_dot(t)
            q_ddot = self.q_ddot(t)
        else:
            q = self.q(self.duration)
            q_dot = np.zeros_like(q)
            q_ddot = np.zeros_like(q)
        policy_state[0] += 1
        action = np.stack([q, q_dot, q_ddot], axis=-2) 
        action = torch.tensor(action, dtype=torch.float32)
        return action, torch.tensor(policy_state)

    def extract_qt(self, x):
        # TODO: make it suitable for parallel envs
        if len(x.shape) == 1:
            x = x[None]
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
            qN = torch.tensor(qN)
            qdN = torch.tensor(qdN)
            qddN = torch.tensor(qddN) 
            tN = torch.tensor(tN)
            tdN = torch.tensor(tdN)

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
