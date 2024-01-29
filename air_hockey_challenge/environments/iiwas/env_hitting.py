from enum import Enum
import numpy as np

from air_hockey_challenge.environments.iiwas.env_double import AirHockeyDouble

class AbsorbType(Enum):
    NONE = 0
    GOAL = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    BOTTOM = 5

class AirHockeyHit(AirHockeyDouble):
    """
        Class for the air hockey hitting task.
    """
    def __init__(self, opponent_agent=None, gamma=0.99, horizon=500, moving_init=True, viewer_params={}):
        """
            Constructor
            Args:
                opponent_agent(Agent, None): Agent which controls the opponent
                moving_init(bool, False): If true, initialize the puck with inital velocity.
        """
        self.hit_range = np.array([[-0.65, -0.25], [-0.4, 0.4]])  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame

        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.moving_init = moving_init
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2
        self.hit_range = np.array([[-0.7, -0.2], [-hit_width, hit_width]])  # Table Frame
        #self.hit_range = np.array([[-0.6, -0.4], [-0.1, 0.1]])  # Table Frame
        #self.hit_range = np.array([[-0.5, -0.5], [0.0, 0.0]])  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        self.absorb_type = AbsorbType.NONE
        self.has_hit = False
        self.hit_time = None
        self.puck_velocity = None

        if opponent_agent is not None:
            self._opponent_agent = opponent_agent.draw_action
        else:
            self._opponent_agent = lambda obs: np.zeros(7)
        self.vy = 1.0
        self.vx = 1.0
        self.v = 0.4
        self.theta = 0.0

    def setup(self, obs):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self.absorb_type = AbsorbType.NONE
        self.has_hit = False
        self.hit_time = None
        #self._write_data("puck_x_pos", -0.5)
        #self._write_data("puck_y_pos", 0.0)
        #self._write_data("puck_x_vel", self.v * np.cos(self.theta))
        #self._write_data("puck_y_vel", self.v * np.sin(self.theta))
        ##self.vy -= 0.2
        ##self.vx += 0.2
        #self.v += 0.2

        if self.moving_init:
            lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
            puck_vel = np.zeros(3)
            puck_vel[0] = -np.cos(angle) * lin_vel
            puck_vel[1] = np.sin(angle) * lin_vel
            puck_vel[2] = np.random.uniform(-2, 2, 1)

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyHit, self).setup(obs)
    
    def _create_info_dictionary(self, obs):
        return {"has_hit": self.has_hit, "hit_time": self.hit_time, "puck_velocity": self.puck_velocity}
    
    def _has_hit(self, state):
        ee_pos, ee_vel = self.get_ee()
        puck_cur_pos, puck_cur_vel = self.get_puck(state)
        # todo: make it reliable for moving puck case
        hit = np.linalg.norm(puck_cur_vel) > 1e-2
        if hit:
            self.hit_time = self._data.time
        return hit
        #if np.linalg.norm(ee_pos[:2] - puck_cur_pos[:2]) < mdp.env_info['puck']['radius'] + \
        #        mdp.env_info['mallet']['radius'] + 5e-3 and np.abs(ee_pos[2] - 0.065) < 0.02:
        #    return True
        #else:
        #    return False

    def reward(self, state, action, next_state, absorbing):
        r = 0
        # Get puck's position and velocity (The position is in the world frame, i.e., center of the table)
        puck_pos, puck_vel = self.get_puck(next_state)
        self.puck_velocity = np.linalg.norm(puck_vel[:2])

        # Define goal position
        goal = np.array([0.98, 0])
        # Compute the vector that shoot the puck directly to the goal
        vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])

        # width of table minus radius of puck
        effective_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius']

        # Calculate bounce point by assuming incoming angle = outgoing angle
        w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[0] -
             effective_width * goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)
        side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])

        # Compute the vector that shoot puck with a bounce to the wall
        vec_puck_side = (side_point - puck_pos[:2]) / \
                        np.linalg.norm(side_point - puck_pos[:2])

        if not self.has_hit:
            self.has_hit = self._has_hit(state)

        #if absorbing or mdp._data.time < mdp.env_info['dt'] * 2:
        if absorbing:
            # If the hit scores
            actual_time = self._data.time
            end_time = self.info.dt * self.info.horizon
            time_left = end_time - actual_time
            energy = np.sum(np.square(puck_vel[:2]))
            energy_integral = energy * time_left
            energy_integral = 100.
            if self.absorb_type == AbsorbType.GOAL:
                r = energy_integral
            elif self.absorb_type == AbsorbType.UP or self.absorb_type == AbsorbType.BOTTOM:
                reward_sign = 1. if self.absorb_type == AbsorbType.UP else -1.
                miss_dist = np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2 
                mul = np.maximum(miss_dist, 0)
                r = reward_sign * energy_integral / (1 + 10. * mul)
            elif self.absorb_type == AbsorbType.LEFT or self.absorb_type == AbsorbType.RIGHT:
                sign = 1. if self.absorb_type == AbsorbType.LEFT else -1.
                goal_x = goal[0] if puck_vel[0] > 0. else -goal[0]
                reward_sign = 1. if puck_vel[0] > 0. else -1.
                time2goalx = (goal_x - puck_pos[0]) / (puck_vel[0] + 1e-8)
                yend = puck_pos[1] + sign * np.abs(puck_vel[1]) * time2goalx
                miss_dist = np.abs(yend) - self.env_info['table']['goal_width'] / 2 
                mul = np.maximum(miss_dist, 0)
                r = reward_sign * energy_integral / (1 + 10. * mul)
            else:
                print(self.absorb_type)
                r = 0.
            self.has_hit = False
        else:
            # If the puck has not yet been hit, encourage the robot to get closer to the puck
            if not self.has_hit:
                ee_pos = self.get_ee()[0][:2]
                dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)
                r = - dist_ee_puck
            else:
                #v = np.sum(np.square(puck_vel[:2]))
                v = np.linalg.norm(puck_vel[:2])

                if puck_vel[0] > 0.:
                    time2goalx = (goal[0] - puck_pos[0]) / (puck_vel[0] + 1e-8)
                    yend = puck_pos[1] + puck_vel[1] * time2goalx
                    miss_dist = np.abs(yend) - self.env_info['table']['goal_width'] / 2 
                    mul = np.maximum(miss_dist, 0)
                    r = v / (1. + 10. * mul)
                else:
                    # TODO think what if the puck goes wrong direction
                    r = puck_vel[0]
        return r

    #def is_absorbing(self, obs):
    #    puck_pos, puck_vel = self.get_puck(obs)
    #    # Stop if the puck bounces back on the opponents wall
    #    if puck_pos[0] > 0 and puck_vel[0] < 0:
    #        return True
    #    return super(AirHockeyHit, self).is_absorbing(obs)
    
    def is_absorbing(self, obs):
        puck_pos = obs[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = obs[3:5].copy()

        if puck_pos[0] < 0.58 or (puck_pos[0] < 0.63 and puck_vel[0] > 0.):
            self.absorb_type = AbsorbType.BOTTOM
            return True

        if puck_pos[0] > 2.46 or (puck_pos[0] > 2.39 and puck_vel[0] < 0):
            self.absorb_type = AbsorbType.UP
            if abs(puck_pos[1]) < 0.1:
                self.absorb_type = AbsorbType.GOAL
            return True

        #if puck_vel[0] > 0. and puck_pos[0] > 1.51:
        if (puck_pos[1] > 0.45 and puck_vel[1] > 0.) or (puck_pos[1] > 0.42 and puck_vel[1] < 0.):
            self.absorb_type = AbsorbType.LEFT
            return True
        if (puck_pos[1] < -0.45 and puck_vel[0] < 0.) or (puck_pos[1] < -0.42 and puck_vel[1] > 0.):
            self.absorb_type = AbsorbType.RIGHT
            return True
        return False

    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        return np.split(obs, 2)[0]

    def _preprocess_action(self, action):
        opponents_obs = np.split(super()._modify_observation(self._obs), 2)[1]

        return action, self._opponent_agent(opponents_obs)


if __name__ == '__main__':
    env = AirHockeyHit(moving_init=True)
    env.reset()

    steps = 0
    while True:
        action = np.zeros(7)

        observation, reward, done, info = env.step(action)
        env.render()
        if done or steps > env.info.horizon:
            steps = 0
            env.reset()
