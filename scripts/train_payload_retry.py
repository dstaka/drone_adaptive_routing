#!/usr/bin/env python
"""Matched retry learning, blocked until independently reconstructed qualification.

No Genesis, Torch or RSL import/GPU initialization occurs before the admission
function accepts actual current source/artifact bindings and independently
replays every required raw case. Status strings cannot bypass that replay.
"""

import argparse
import copy
import json
import pickle
from pathlib import Path

from adaptive_drone_patrol.payload_checkpoint import validate_parent_config, actor_signature
from adaptive_drone_patrol.payload_retry_checkpoint import transfer_state, initialization_comparison
from adaptive_drone_patrol.payload_retry_contract import OBSERVATION_CONTRACT, PROFILE
from adaptive_drone_patrol.payload_retry_admission import require_retry_admission
from adaptive_drone_patrol.util import sha256, json_write

PARENT_SHA256 = '26658a31560f71c36fd9e9605b15d8e4f7b9cae8174c1465fd12dcd0d10b3431'
CONFIG_SHA256 = '35c6b0be8db90e4e799d453d447a484275317f85098a22cbdf04c19c84056e55'
SOURCES = (
    'scripts/train_payload_retry.py',
    'scripts/probe_payload_ball.py',
    'scripts/probe_payload_ball_bank.py',
    'src/adaptive_drone_patrol/payload_retry_contract.py',
    'src/adaptive_drone_patrol/payload_retry_checkpoint.py',
    'src/adaptive_drone_patrol/payload_retry_training.py',
    'src/adaptive_drone_patrol/payload_training.py',
    'src/adaptive_drone_patrol/payload_contract.py',
    'src/adaptive_drone_patrol/payload_checkpoint.py',
    'src/adaptive_drone_patrol/payload_cases.py',
    'src/adaptive_drone_patrol/payload_ball_bank.py',
    'src/adaptive_drone_patrol/payload_ball_asset.py',
    'src/adaptive_drone_patrol/payload_probe_controller.py',
    'src/adaptive_drone_patrol/payload_qualification.py',
    'src/adaptive_drone_patrol/payload_substep_probe.py',
    'src/adaptive_drone_patrol/payload_retry_admission.py',
    'src/adaptive_drone_patrol/payload_retry_properties.py',
    'src/adaptive_drone_patrol/util.py',
    'configs/payload_retry_probe_manifest.json',
    'configs/payload_retry_motor_audit.json',
    'requirements/flight-linux.lock.txt',
)


def training_configuration(configs, method):
    if method not in ('randomized', 'fixed') or configs[4].get('num_steps_per_env') != 100:
        raise ValueError('Matched retry learning requires 100 rollout steps and a declared method')
    train = copy.deepcopy(configs[4])
    train.update(run_name=method, save_interval=500)
    train['algorithm'].update(learning_rate=.00003, schedule='fixed', entropy_coef=0., max_grad_norm=.3)
    return train


def actual_training_binding(root, parent, config):
    root = Path(root)
    declaration = json.loads((root/'configs/payload_retry_probe_manifest.json').read_text())
    if declaration.get('profile') != PROFILE:
        raise ValueError('Retry training requires the explicit root-aligned-v3 declaration')
    return dict(profile=PROFILE, observation_contract=copy.deepcopy(OBSERVATION_CONTRACT),
                source_binding={name: sha256(root/name) for name in SOURCES},
                manifest_sha256=sha256(root/'configs/payload_retry_probe_manifest.json'),
                motor_plan_sha256=sha256(root/'configs/payload_retry_motor_audit.json'),
                backend_source_sha256=sha256(root/'src/adaptive_drone_patrol/payload_ball_bank.py'),
                parent_sha256=sha256(parent), parent_config_sha256=sha256(config),
                control_dt_s=.01, rigid_dt_s=.005, requested_root_alignment=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--admission', required=True, type=Path)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--reconstruction', required=True, type=Path)
    parser.add_argument('--reconstruction-sha256', required=True)
    parser.add_argument('--method', choices=('randomized', 'fixed'), required=True)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError('A positive matched iteration budget is required')
    if args.parent_sha256 != PARENT_SHA256 or args.config_sha256 != CONFIG_SHA256:
        raise ValueError('Explicit hashes must identify adopted model4296 and its actual configuration')
    if args.out.exists():
        raise FileExistsError('Retry output already exists; preserve previous attempts')
    if sha256(args.admission) != args.admission_sha256 or sha256(args.reconstruction) != args.reconstruction_sha256:
        raise ValueError('Qualification/reconstruction receipt hash mismatch')
    admission = json.loads(args.admission.read_text())
    reconstructed = json.loads(args.reconstruction.read_text())
    binding = actual_training_binding(Path.cwd(), args.parent, args.config)
    # Reconstruct actual raw evidence with the current physical and checker sources.
    admitted = require_retry_admission(admission, binding, reconstructed)
    binding['physical_signature_sha256'] = admitted['physical_signature_sha256']
    if binding['parent_sha256'] != args.parent_sha256 or binding['parent_config_sha256'] != args.config_sha256:
        raise ValueError('Actual adopted parent/configuration bytes differ')
    with args.config.open('rb') as stream:
        configs = pickle.load(stream)
    validate_parent_config(configs)
    train = training_configuration(configs, args.method)
    effective_training = copy.deepcopy(train)
    budget = args.iterations*train['num_steps_per_env']
    import genesis as gs
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from adaptive_drone_patrol.payload_retry_training import PayloadRetryTrainingEnv
    args.out.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision='32', logging_level='warning', seed=7)
    env = PayloadRetryTrainingEnv(args.out/'scenes', args.method == 'randomized', budget, seed=7)
    runner = OnPolicyRunner(env, train, str(args.out), device=gs.device)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    initial = transfer_state(parent)
    comparison = initialization_comparison(parent, initial)
    torch.save(initial, args.out/'initialization.pt')
    load_cfg = dict(actor=True, critic=True, optimizer=False, iteration=False, rnd=False)
    runner.load(str(args.out/'initialization.pt'), load_cfg=load_cfg)
    actual_actor_hash = actor_signature(runner.get_inference_policy(device=gs.device).state_dict())
    if actual_actor_hash != comparison['initial_actor_tensor_sha256']:
        raise ValueError('Actual initialized actor differs from the explicit tensor transfer')
    manifest = dict(
        schema='payload-retry-learning-v1', status='TRAINING_IN_PROGRESS_NOT_VALIDATED',
        profile=PROFILE, method=args.method, seed=7, num_envs=1, environment_steps=budget,
        num_steps_per_env=100, iterations=args.iterations, actual_environment_steps_at_start=env.total_steps,
        observation_contract=copy.deepcopy(OBSERVATION_CONTRACT),
        parent_sha256=binding['parent_sha256'], parent_config_sha256=binding['parent_config_sha256'],
        source_binding=binding['source_binding'], actual_execution_binding=binding,
        optimizer_config=effective_training['algorithm'],
        architecture={key: effective_training[key] for key in ('actor', 'critic', 'obs_groups')},
        admission_sha256=sha256(args.admission), reconstruction_sha256=sha256(args.reconstruction),
        initialization_sha256=sha256(args.out/'initialization.pt'), initialization_comparison=comparison,
        actual_initial_actor_tensor_sha256=actual_actor_hash, checkpoint_load_options=load_cfg,
        fresh_optimizer=True, initial_iteration=0,
        curriculum=['hover', 'short', 'medium', 'long', 'reversal/ascent/descent', 'dispersed_xyz'],
        full_distribution_every_stage=True,
        sampling_contract=dict(randomized='Every episode: independent uniform ratio0.10–0.30 and equiprobable side0.030/0.045/0.060',
                               fixed='Every episode: ratio0.20, side0.045',
                               payload_stream=[7, 710, 0], pose_stream=[7, 710, 101], command_stream=[7, 710, 102]),
        reward_implementation='Unchanged PayloadTrainingEnv.step reward/components and pre-reset terminal guards',
        terminal_evidence='terminal-states.jsonl; invalid-raw-state.json before abort',
        heldout_opened=False, operational_acceptance='NOT_RUN',
    )
    json_write(args.out/'training_manifest.json', manifest)
    with (args.out/'parent_cfgs.pkl').open('xb') as stream:
        pickle.dump(configs, stream)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    if env.total_steps != budget:
        raise RuntimeError('Actual training environment-step budget is incomplete or differs')
    runner.save(str(args.out/'final.pt'), infos={'observation_contract': OBSERVATION_CONTRACT,
                'physics_signature': PROFILE, 'status': 'TRAINED_NOT_SELECTED'})
    json_write(args.out/'completed.json', dict(checkpoint_sha256=sha256(args.out/'final.pt'),
               training_manifest_sha256=sha256(args.out/'training_manifest.json'),
               actual_environment_steps=env.total_steps, episodes=env.episode_count,
               failures=env.failures, completed_training_stops=env.completed_stops,
               reward_components=env.reward_totals, heldout_opened=False,
               status='TRAINED_NOT_SELECTED', operational_acceptance='NOT_RUN'))


if __name__ == '__main__':
    main()
