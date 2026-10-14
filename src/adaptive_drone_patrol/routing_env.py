"""Gymnasium analytical routing environment; Genesis is never imported here."""

import gymnasium as gym
import numpy as np
from .scenario import generate_scenario
from .objective import RoutingState, SCHEMA


class RoutingEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenario=None, event_mixture=False, max_steps=None):
        self.fixed = scenario
        self.event_mixture = event_mixture
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (SCHEMA['size'],), np.float32)
        self.state = None
        self.steps = 0
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps, self.finished = 0, False
        kind = self.np_random.choice(['static', 'priority', 'cost'], p=[.5, .25, .25]) \
            if self.event_mixture else 'static'
        scenario = self.fixed or generate_scenario(int(self.np_random.integers(10000)), 'training', kind)
        self.state = RoutingState(scenario)
        return self.state.observation(), {}

    def action_masks(self):
        return self.state.mask()

    def step(self, action):
        if self.finished:
            raise RuntimeError('Cannot step a terminal or truncated environment; reset first')
        action = int(action)
        self.steps += 1
        if action not in range(3) or self.state.visited[action]:
            self.state.invalid_count += 1
            truncated = self.state.invalid_count >= 3 or bool(self.max_steps and self.steps >= self.max_steps)
            self.finished = truncated
            return self.state.observation(), 0., False, truncated, {'invalid_action': True}
        snapshot = self.state.departure(action)
        cost = self.state.complete(action, snapshot)
        event = self.state.apply_event()
        terminated = self.state.terminal
        truncated = bool(not terminated and self.max_steps and self.steps >= self.max_steps)
        self.finished = terminated or truncated
        return self.state.observation(), -cost, terminated, truncated, {'leg': snapshot, 'event': event}
