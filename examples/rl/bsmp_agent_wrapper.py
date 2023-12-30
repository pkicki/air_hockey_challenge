import numpy as np

from air_hockey_challenge.framework import AgentBase
from mushroom_rl.core import Agent

class BSMPAgent(AgentBase):
    def __init__(self, env_info, bsmp_agent: Agent):
        self.env_info = env_info
        self.bsmp_agent = bsmp_agent
        super().__init__(env_info, is_episodic=True)

        self._add_save_attr(
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
        return self.bsmp_agent.draw_action(state, policy_state)

    def episode_start(self, initial_state, episode_info):
        return self.bsmp_agent.episode_start(initial_state, episode_info)

    def episode_start_vectorized(self, initial_states, episode_info, start_mask):
        policy_states, theta = self.bsmp_agent.compute_trajectory(initial_states)
        for p in policy_states:
            p['iteration'] = 0
        return np.array(policy_states), np.array(theta)

    def fit(self, dataset, **info):
        self.bsmp_agent.fit(dataset, **info)

    def update_alphas(self):
        self.bsmp_agent.update_alphas()

    def get_alphas(self):
        return self.bsmp_agent.alphas

    def preprocess(self, state):
        return state