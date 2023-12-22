import time

import numpy as np

from mushroom_rl.core import Core, VectorCore


class ChallengeCoreVectorized(VectorCore):
    """
    Wrapper for mushrooms Core. Used to time the agents draw_action function and select indices of actions.
    """
    def __init__(self, *args, action_idx=None, is_tournament=False, time_limit=None, init_state=None, **kwargs):
        """
        Constructor.

        Args:
            action_idx(list, None): Indices of action which should get used. Default is the first n indices where n is
                the action length
            is_tournament(bool, False): Flag that is set when environment is tournament. Needed because the tournament
                Agent keeps track of the time
            time_limit(float, None): Time limit for tournament environment. If draw_action took limit than time_limit
                the previous action is reused
            init_state(list, None): Initial state of the robot. Used as initial value for previous action
        """
        super().__init__(*args, **kwargs)

        if action_idx:
            self.action_idx = action_idx
        else:
            if is_tournament:
                self.action_idx = (np.arange(self.env.base_env.action_shape[0][0]), np.arange(self.env.base_env.action_shape[1][0]))
            else:
                self.action_idx = np.arange(self.env.base_env.action_shape[0][0])

        self.is_tournament = is_tournament
        self.time_limit = time_limit

        self.prev_action = None
        self.init_state = init_state

    def _step(self, render, record, mask):
        """
        Single step.

        Args:
            render (bool):
                whether to render or not.
            record (bool):
                whether to record the rendered frame or not.

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        # TODO: Adjust the tournament part of the challenge core to align with the vectorized core and policy_states
        if self.is_tournament:
            action_1, action_2, time_1, time_2 = self.agent.draw_action(self._state)

            if self.time_limit:
                if time_1 > self.time_limit:
                    action_1 = self.prev_action[0]
                    action_1[1] = 0

                if time_2 > self.time_limit:
                    action_2 = self.prev_action[1]
                    action_2[1] = 0

            self.prev_action = [action_1.copy(), action_2.copy()]
            action = (action_1, action_2)
            duration = [time_1, time_2]

            next_state, reward, absorbing, step_info = self.env.step(
                (action[0][self.action_idx[0]], action[1][self.action_idx[1]]))

        else:
            start_time = time.time()
            action, policy_next_state = self.agent.draw_action(self._state, self._policy_state)
            end_time = time.time()
            duration = (end_time - start_time)

            # If there is an index error here either the action shape does not match the interpolation type or
            # the custom action_idx is wrong
            next_state, rewards, absorbing, step_info = self.env.step_all(mask, action[:, self.action_idx])

        for i, s in enumerate(step_info):
            if mask[i]:
                s["computation_time"] = duration
        self._episode_steps[mask] += 1

        if render:
            frames = self.env.render_all(mask, record=record)

            if record:
                for i in range(self.env.number):
                    if mask[i]:
                        self._record[i](frames[i])

        last = absorbing | (self._episode_steps >= self.env.info.horizon)

        state = self._state
        policy_state = self._policy_state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state
        self._policy_state = policy_next_state

        # Hotfix for the action storage in the dataset for not tournament environment
        # Definitely will not work for the tournament environment
        # TODO: Fix this by adding a different type of the dataset or by proper reshaping of the action
        # and reporting flattened shapes while creating enviornment
        action = np.reshape(action[:, self.action_idx], (action.shape[0], -1))

        return (state, action, rewards, next_state, absorbing, last, policy_state, policy_next_state), step_info

    def reset(self, initial_states=None):
        super().reset(initial_states)
        self.prev_action = (np.vstack([self.init_state, np.zeros_like(self.init_state)]),
                            np.vstack([self.init_state, np.zeros_like(self.init_state)]))
