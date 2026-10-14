import numpy as np
import pytest

from adaptive_drone_patrol.routing_agent import RoutingAgent, train_one
from adaptive_drone_patrol.scenario import generate_scenario
from adaptive_drone_patrol.objective import RoutingState


def test_real_masked_ppo_save_reload_and_terminal_guard(tmp_path):
    path = train_one(42, tmp_path / 'model', static_steps=256, event_steps=256, n_epochs=1)
    agent = RoutingAgent(path)
    state = RoutingState(generate_scenario(20001, 'static'))
    while not state.terminal:
        action = agent.predict(state)
        assert state.mask()[action]
        assert np.isfinite(state.observation()).all()
        state.complete(action, state.departure(action))
    with pytest.raises(RuntimeError, match='terminal'):
        agent.predict(state)
