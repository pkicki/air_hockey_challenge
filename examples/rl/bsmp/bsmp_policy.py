import numpy as np

from mushroom_rl.policy import Policy


class BSMPPolicy(Policy):
    def __init__(self, dt):
        self._dt = dt
        policy_state_shape = tuple()
        super().__init__(policy_state_shape)

        self._add_save_attr(
            _dt='primitive',
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
        if not isinstance(policy_state, np.ndarray):
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
