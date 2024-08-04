"""Fastest pong in the west"""

import numpy as np

class VectorizedPongEnv:
    def __init__(self, n_envs=1):
        self.n_envs = n_envs
        self.width, self.height = 800, 600
        self.paddle_width, self.paddle_height = 10, 100
        self.ball_radius = 10
        self.paddle_speed = 10
        self.ball_speed = 7
        self.reset()

    def reset(self):
        self.paddle1_y = np.full(self.n_envs, self.height / 2 - self.paddle_height / 2)
        self.paddle2_y = np.full(self.n_envs, self.height / 2 - self.paddle_height / 2)
        self.ball_x = np.full(self.n_envs, self.width / 2)
        self.ball_y = np.full(self.n_envs, self.height / 2)
        self.ball_dx = np.full(self.n_envs, self.ball_speed)
        self.ball_dy = np.random.uniform(-self.ball_speed, self.ball_speed, self.n_envs)
        self.scores = np.zeros((self.n_envs, 2), dtype=int)
        self.masks = np.ones(self.n_envs, dtype=bool)
        return self._get_obs()

    def step(self, actions):
        # Update paddle positions
        self.paddle1_y[self.masks] += actions[self.masks] * self.paddle_speed
        self.paddle1_y[self.masks] = np.clip(self.paddle1_y[self.masks], 0, self.height - self.paddle_height)
        
        # Smooth AI for paddle2
        paddle2_target = self.ball_y[self.masks] - self.paddle_height / 2
        self.paddle2_y[self.masks] += np.clip(paddle2_target - self.paddle2_y[self.masks], -self.paddle_speed, self.paddle_speed)
        self.paddle2_y[self.masks] = np.clip(self.paddle2_y[self.masks], 0, self.height - self.paddle_height)

        # Update ball position
        self.ball_x[self.masks] += self.ball_dx[self.masks]
        self.ball_y[self.masks] += self.ball_dy[self.masks]

        # Ball collision with top and bottom walls
        bounce_y = (self.ball_y <= self.ball_radius) | (self.ball_y >= self.height - self.ball_radius)
        self.ball_dy[bounce_y & self.masks] *= -1
        self.ball_y[self.masks] = np.clip(self.ball_y[self.masks], self.ball_radius, self.height - self.ball_radius)

        # Ball collision with paddles
        bounce_left = (self.ball_x <= self.paddle_width + self.ball_radius) & \
                      (self.ball_y >= self.paddle1_y) & (self.ball_y <= self.paddle1_y + self.paddle_height)
        bounce_right = (self.ball_x >= self.width - self.paddle_width - self.ball_radius) & \
                       (self.ball_y >= self.paddle2_y) & (self.ball_y <= self.paddle2_y + self.paddle_height)
        
        self.ball_dx[(bounce_left | bounce_right) & self.masks] *= -1

        # Scoring
        score_left = self.ball_x >= self.width
        score_right = self.ball_x <= 0
        self.scores[:, 0] += score_left & self.masks
        self.scores[:, 1] += score_right & self.masks

        # Determine rewards and done state
        rewards = np.zeros(self.n_envs)
        rewards[self.masks] = score_left[self.masks].astype(float) - score_right[self.masks].astype(float)
        done = (score_left | score_right) & self.masks

        # Update masks
        self.masks[done] = False

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        obs = np.column_stack([
            self.paddle1_y / self.height,
            self.paddle2_y / self.height,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / self.ball_speed,
            self.ball_dy / self.ball_speed
        ])
        return obs
