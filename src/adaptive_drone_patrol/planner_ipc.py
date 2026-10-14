"""Persistent CPU planner protocol with bounded, fail-closed JSON exchange."""

import argparse
import json
import os
import select
import subprocess
import sys
import time
import numpy as np
from .objective import SCHEMA, schema_for


class PlannerClient:
    def __init__(self, command, policy_hash, timeout=10, schema=None):
        self.schema = schema or SCHEMA
        self.n = self.schema.get('num_checkpoints', 3)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=sys.stderr)
        self.policy_hash, self.timeout = policy_hash, timeout
        self.request_id = 0
        self.buffer = b''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()

    def request(self, features, mask):
        features_array, mask_array = np.asarray(features), np.asarray(mask)
        if (features_array.shape != (self.schema['size'],) or mask_array.shape != (self.n,) or
                not np.isfinite(features_array).all() or mask_array.dtype != np.dtype(bool)):
            raise ValueError('Invalid IPC dimensions or values; retrain for the required contract')
        if not mask_array.any():
            raise RuntimeError('No terminal policy query is allowed')
        self.request_id += 1
        q = {'request_id': self.request_id, 'schema': self.schema['id'], 'num_checkpoints': self.n,
             'checkpoint_ids': self.schema.get('checkpoint_ids', ['A', 'B', 'C']), 'features': list(features),
             'mask': list(map(bool, mask)), 'policy_hash': self.policy_hash}
        try:
            self.process.stdin.write((json.dumps(q, allow_nan=False) + '\n').encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError('Planner process died') from exc
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Planner timeout')
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise RuntimeError('Planner timeout')
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError('Planner process died')
            self.buffer += block
            if len(self.buffer) > 1024 * 1024:
                raise RuntimeError('Planner response too large')
        line, self.buffer = self.buffer.split(b'\n', 1)
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError('Malformed planner response') from exc
        if type(r.get('request_id')) is not int or r['request_id'] != self.request_id:
            raise RuntimeError('Planner request ID mismatch')
        if r.get('status') != 'OK' or r.get('policy_hash') != self.policy_hash:
            raise RuntimeError('Planner failed or policy hash mismatch')
        action = r.get('action')
        if type(action) is not int or action not in range(self.n) or not mask[action]:
            raise RuntimeError('Planner invalid action')
        return action

    def predict(self, state):
        if (state.scenario.num_checkpoints != self.n or
                (self.schema != SCHEMA and schema_for(state.scenario) != self.schema) or
                (self.schema == SCHEMA and (state.scenario.checkpoint_ids != ['A', 'B', 'C'] or
                 state.scenario.alpha != 1. or state.scenario.beta != 1. or
                 state.scenario.nominal_speed_mps != 1. or state.scenario.return_to_base))):
            raise ValueError('Planner/scenario mismatch; retrain for this contract')
        return self.request(state.observation(legacy=self.schema == SCHEMA).tolist(), state.mask().tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--hash', required=True)
    p.add_argument('--validate-scenario')
    a = p.parse_args()
    from .routing_agent import RoutingAgent
    agent = RoutingAgent(a.model, a.hash)
    if a.validate_scenario:
        from .scenario import Scenario
        from .util import load_config
        scenario = Scenario.from_dict(load_config(a.validate_scenario))
        agent.validate_scenario(scenario)
        print(json.dumps({'status': 'OK', 'schema': agent.schema, 'policy_hash': agent.hash}))
        return
    for line in sys.stdin:
        request = json.loads(line)
        result = {'request_id': request.get('request_id'), 'policy_hash': agent.hash}
        try:
            if request['schema'] != agent.schema['id'] or request['policy_hash'] != agent.hash:
                raise ValueError('Protocol schema/policy mismatch')
            if request.get('num_checkpoints') != agent.num_checkpoints or request.get('checkpoint_ids') != agent.schema.get('checkpoint_ids', ['A', 'B', 'C']):
                raise ValueError('Protocol dimensions/ID ordering mismatch; retrain for this contract')
            features = np.asarray(request['features'], dtype=np.float32)
            action = agent.predict_features(features, request['mask'])
            result.update(status='OK', action=action)
        except Exception as exc:
            result.update(status='FAILED', error=str(exc))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
