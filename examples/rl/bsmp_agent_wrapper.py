import numpy as np

from air_hockey_challenge.framework import AgentBase
from mushroom_rl.core import Agent

class DummyPolicy:
    def __init__(self):
        pass

    def compute_action_and_log_prob(self, state):
        return None, np.zeros((1,))

class DummyRLAgent(AgentBase):
    def __init__(self):
        self.policy = DummyPolicy()
        self._alpha = 0.

    def _critic_approximator(self, s, a):
        return np.zeros((1,))

class BSMPAgent(AgentBase):
    def __init__(self, env_info, bsmp_agent: Agent):
        self.env_info = env_info
        self.bsmp_agent = bsmp_agent
        super().__init__(env_info)

        self._dt = env_info["dt"]

        self._pos_limit = env_info["robot"]["joint_pos_limit"]
        self._vel_limit = env_info["robot"]["joint_vel_limit"]
        self._acc_limit = env_info["robot"]["joint_acc_limit"]

        self._planner_calls = 0
        self._iteration = 0
        self.q = None
        self.q_dot = None
        self.q_ddot = None

        self.theta_list = []
        self.q_log_t_cps_mu_trainable_list = []
        self.q_log_t_cps_mu_list = []
        self.distribution_list = []

        self.rl_agent = DummyRLAgent()

        self._add_save_attr(
            _dt='primitive',
            _pos_limit='numpy',
            _vel_limit='numpy',
            _acc_limit='numpy',
            _init_pos='numpy',
            rl_agent='pickle',
            env_info='pickle'
        )

    def draw_action(self, state):
        """
        Args:
            state (ndarray): state of the system

        Returns:
            numpy.ndarray, (3, num_joints): The desired [Positions, Velocities, Acceleration] of the
            next step. The environment will take first two arguments of the to control the robot.
            The third array is used for the training of the SAC as the output is acceleration. This
            action tuple will be saved in the dataset buffer
        """
        #print("Iteration: ", self._iteration)
        if self._iteration == 0:
            print("Trajectory computation: ", self._planner_calls)
            trajectory = self.bsmp_agent.compute_trajectory(state)
            self.q = trajectory["q"]
            self.q_dot = trajectory["q_dot"]
            self.q_ddot = trajectory["q_ddot"]
            self.duration = trajectory["duration"]
            self.theta_list.append(trajectory["theta"])
            self.q_log_t_cps_mu_trainable_list.append(trajectory['mu_trainable'])
            self.q_log_t_cps_mu_list.append(trajectory['mu'])
            self.distribution_list.append(trajectory['distribution'])
            self._planner_calls += 1
        
        time = min(self._iteration * self._dt, self.duration)
        pos = self.q(time)
        vel = self.q_dot(time)
        acc = self.q_ddot(time)
        self._iteration += 1
        return np.vstack([pos, vel, acc])

    def reset(self):
        self._iteration = 0
        self.q = None
        self.q_dot = None
        self.q_ddot = None

    def reset_dataset(self):
        self.theta_list = []
        self.q_log_t_cps_mu_trainable_list = []
        self.q_log_t_cps_mu_list = []
        self.distribution_list = []
        

    def fit(self, dataset, **info):
        self.bsmp_agent.fit(dataset, self.theta_list, self.q_log_t_cps_mu_trainable_list,
                            self.q_log_t_cps_mu_list, self.distribution_list, **info)
        self.reset_dataset()

    def update_alphas(self):
        self.bsmp_agent.update_alphas()

    def preprocess(self, state):
        return state