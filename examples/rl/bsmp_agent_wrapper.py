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
        super().__init__(env_info, is_episodic=True)

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
            _planner_calls='primitive',
            _iteration='primitive',
            rl_agent='pickle',
            bsmp_agent='mushroom',
            env_info='pickle'
        )

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

        single_env = False
        if not isinstance(policy_state, list):
            single_env = True
            policy_state = [policy_state]
        q = []
        q_dot = []
        q_ddot = []
        for ps in policy_state:
            t = min(ps["iteration"] * self._dt, ps["duration"])
            q.append(ps["q"](t))
            q_dot.append(ps["q_dot"](t))
            q_ddot.append(ps["q_ddot"](t))
            ps["iteration"] += 1
        action = np.stack([np.array(x) for x in [q, q_dot, q_ddot]], axis=-2) 

        if single_env:
            action = action[0]
            policy_state = policy_state[0]

        return action, policy_state

    def episode_start(self, initial_state, episode_info):
        policy_states, theta = self.bsmp_agent.compute_trajectory(initial_state[None])
        policy_states[0]['iteration'] = 0
        return policy_states[0], theta[0]

    def episode_start_vectorized(self, initial_states, episode_info, start_mask):
        policy_states, theta = self.bsmp_agent.compute_trajectory(initial_states)
        for p in policy_states:
            p['iteration'] = 0
        return policy_states, theta

    def fit(self, dataset, **info):
        self.bsmp_agent.fit(dataset, **info)

    def update_alphas(self):
        self.bsmp_agent.update_alphas()

    def get_alphas(self):
        return self.bsmp_agent.alphas

    def preprocess(self, state):
        return state