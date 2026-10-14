"""Persistent CPU planner protocol with bounded, fail-closed JSON exchange."""

import argparse
import json
import os
import select
import subprocess
import sys
import time
import numpy as np
from .objective import SCHEMA


class PlannerClient:
    def __init__(self, command, policy_hash, timeout=10):
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
        self.request_id += 1
        q = {'request_id': self.request_id, 'schema': SCHEMA['id'], 'features': list(features),
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
        if r.get('request_id') != self.request_id:
            raise RuntimeError('Planner request ID mismatch')
        if r.get('status') != 'OK' or r.get('policy_hash') != self.policy_hash:
            raise RuntimeError('Planner failed or policy hash mismatch')
        action = r.get('action')
        if type(action) is not int or action not in range(3) or not mask[action]:
            raise RuntimeError('Planner invalid action')
        return action

    def predict(self, state):
        return self.request(state.observation().tolist(), state.mask().tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--hash', required=True)
    a = p.parse_args()
    from .routing_agent import RoutingAgent
    agent = RoutingAgent(a.model, a.hash)
    for line in sys.stdin:
        request = json.loads(line)
        result = {'request_id': request.get('request_id'), 'policy_hash': agent.hash}
        try:
            if request['schema'] != SCHEMA['id'] or request['policy_hash'] != agent.hash:
                raise ValueError('Protocol schema/policy mismatch')
            features = np.asarray(request['features'], dtype=np.float32)
            action = agent.predict_features(features, request['mask'])
            result.update(status='OK', action=action)
        except Exception as exc:
            result.update(status='FAILED', error=str(exc))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
