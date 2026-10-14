# Adaptive Drone Inspection: Autonomous Implementation Specification

Date: 2026-10-13. Target platform: Vast.ai Linux, NVIDIA RTX 5060 Ti, 16 GB VRAM.

## 1. Execution contract and required outcome

Read this entire file and use it as the complete implementation contract. Autonomously create the code, install dependencies in isolated environments where needed, train or reuse the specified models, run tests, repair failures, collect genuine simulation evidence, prepare the submission artifacts, synchronize the result to the operator's local checkout, and make a local Git commit. A design document or instructions for someone else to implement are not the requested end state.

The operator has authorized implementation, routine reversible fixes, SSH commands on the supplied Vast.ai instance, model training and evaluation within the budget below, artifact transfer, and local Git commits. Do not repeatedly ask for these permissions. Follow actual tool approval requirements. Ask focused questions when missing information prevents progress, and continue independent work while waiting. Do not guess connection details, invent credentials, or infer approval for additional expenditure.

When the operator instructs execution of this file, treat it as the selected design and an explicit request for autonomous planning and implementation. Create a concise implementation checklist/plan and proceed; do not pause merely to request another design or plan approval. Apply Superpowers planning, debugging, testing, and verification practices where available without adding discretionary approval gates. This execution instruction does not replace an actual tool-enforced permission or a required missing-input question.

The task is an adaptive inspection patrol of exactly three checkpoints, A, B, and C. A learned routing policy chooses the next unvisited checkpoint using observable urgency and logical travel costs. A learned flight policy controls a drone in Genesis to execute those choices. Replan only after a physically verified arrival and inspection dwell. Complete all three checkpoints in each of three scenarios: normal inspection, a priority update, and a logical travel-cost update. Record authentic Genesis video for every scenario.

The operator explicitly selected the complete patrol as mandatory. A one-goal flight, a planning-only animation, a partially completed patrol, or separate unintegrated policies is useful intermediate evidence but does not satisfy this specification. Report an unmet requirement as incomplete. Do not redefine success to fit the available results.

Resource limit: use the already rented and authorized instance. The cumulative elapsed time of GPU execution phases must not exceed **8 hours**, including initialization, training, inference, recording, retries, and waiting during an active GPU phase. Never run competing GPU jobs. Persist elapsed time across resumptions in a run ledger. Before each job, check the remaining allowance and enforce a process deadline with orderly checkpoint/video finalization. Stop before the cap, synchronize existing artifacts, and request an extension only if more GPU work is needed. This limit does not authorize renting, extending a rental, changing a billing plan, or terminating an instance. Rental billing can continue while the instance is idle; tell the operator when GPU work has finished.

All text in this specification, all newly authored or adapted source comments, and all code docstrings must be English. Identifiers, CLI options, schemas, and machine-readable status values must also be English. Preserve supplied reference files unchanged. Translate copied non-English comments in generated implementation files. Japanese README prose, report prose, chart labels, and video captions are required for the course audience; these are user-facing content, not code comments.

This specification has no dependency on another specification or a previous conversation. The notebook and challenge images are supporting source artifacts; the relevant requirements and notebook contracts are stated below. If a supporting artifact is unavailable, continue using this file and verify the actual installed source before implementing its adapter.

## 2. Course requirements and submission artifacts

The supplied challenge statements require:

- An original robot task that uses AI in some form. Inference is sufficient; training is optional at the course level.
- A video showing the robot executing its task, lasting **60 to 180 seconds**. Playback acceleration is permitted if explicitly labeled; the statements mention 2x playback as an example.
- All code used for the task.
- A short report explaining the task, purpose, and original design choices.
- Clear presentation, technical substance, originality, and a plausible real-world application as quality criteria. Reused work must be attributed; copying the course sample without an original contribution is unacceptable.

This project adds stricter requirements: learned adaptive routing, three complete physical patrols, arrival-boundary updates, reproducible comparisons, a Japanese report PDF, and a tested local Git commit. The course does not establish an RL superiority requirement, a publication platform, or a due date in the provided images. Do not invent any of those.

Required final artifacts:

| Artifact | Required content |
| --- | --- |
| `README.md` | Japanese setup, remote workflow, verified commands, architecture, results, limitations, regeneration instructions |
| `src/`, `scripts/`, `configs/`, `tests/` | Complete implementation and automated checks |
| Dependency locks | Exact working flight dependencies and independently resolved routing dependencies |
| `docs/proven-baseline.md` | Notebook provenance, remote reproduction, source/checkpoint hashes, environment evidence |
| `docs/compatibility.md` | Verified package versions, API contracts, frame conventions, rendering contract |
| `docs/verification.md` | Commands, exit codes, statuses, logs, artifact references, unmet requirements |
| `docs/report_ja.md` | Short Japanese report source with actual results and original contribution |
| `outputs/report/report.pdf` | Readable Japanese PDF rendered and visually inspected |
| `outputs/evaluation/` | Held-out benchmarks, flight tests, complete scenario logs, metrics, figures |
| `outputs/videos/raw_{normal,emergency,cost_shift}.mp4` | Actual recorded Genesis patrols with matching telemetry |
| `outputs/videos/submission.mp4` | 60-180 second Japanese demonstration, target length 120 seconds |
| `outputs/videos/timeline.json` | Mapping from source simulation times and frames to edited video times |
| `artifacts/manifest.json` | SHA-256, byte size, relative path, run ID, regeneration command for retained artifacts |
| `THIRD_PARTY_NOTICES.md` | Reused sources/assets, exact revisions, licenses, modifications, original contribution |
| Local Git commit | Specification, implementation, tests, configurations, locks, documentation, compact evidence summaries |

Keep large videos, checkpoints, embedded notebook media, caches, and private connection information out of Git by default. They must still be delivered locally and listed in the manifest. Include all source used to generate them. Do not publish a repository, upload a video, or submit to the course unless separately instructed.

## 3. Proven notebook baseline and feasibility assessment

### 3.1 Evidence already supplied

The operator has demonstrated a single-destination flight video on Linux with an RTX 5060 Ti and 16 GB VRAM using `RL_drone_hovering.ipynb`. The supplied notebook contains training and recording code plus saved execution output. This is stronger evidence than an untested upstream example. It establishes an existing PPO training/inference/recording workflow, but does not establish multi-goal patrol reliability or adaptation performance.

Reference hashes at specification authoring:

| File | SHA-256 |
| --- | --- |
| `RL_drone_hovering.ipynb` | `4d993fa3cb9fcb53d77bfa6106b26864316a3c2b70e003e67fbbd6f18b8daf8d` |
| `Challenge_statement1.png` | `cc41910d3c3497cf628a9c5e18672403d8a87cd7af6471ee1f8c3b2aa37e077e` |
| `Challenge_statement2.png` | `f8040c933fad7e063fde73a1576f399bbd34a376491ee6f356dbeeb843c931e8` |

Notebook facts, with their evidence level:

| Item | Recorded value | Evidence |
| --- | --- | --- |
| Python | `3.12.14` | Notebook kernel metadata; installation output uses Python 3.12 paths |
| Genesis package | `genesis-world==1.4.3` | Saved package-version output |
| Example source reference | `v1.4.3` | Code downloads official `examples/drone/{hover_env,hover_train,hover_eval}.py` at this reference |
| RSL-RL | `rsl-rl-lib==5.5.1` | Saved installation output; notebook requests `>=5.0.0,<6` |
| PyTorch | `2.14.0+cu132` | Saved installation output |
| NumPy | `2.5.3` | Saved installation output |
| TensorDict | `0.14.3` | Saved installation output |
| TensorBoard | `2.21.0` | Saved installation output |
| CUDA Python dependency | `cuda-toolkit==13.2.1` | Saved installation output; not a verified host driver version |
| Training | 256 environments, 301 iterations | Code and saved PPO output |
| Policy architecture | 17 inputs, two 128-unit Tanh hidden layers, 4 actor outputs | Saved runner output |
| Checkpoint | `logs/drone-hovering/model_300.pt` | Saved evaluation output |
| Historical working paths | `/drone_hovering`, `/venv/main` | Saved output; discovery hints, not guaranteed paths on a new instance |
| Evaluation | One environment, 1,000 steps, 100 Hz simulation | Code and output reporting 10 simulation seconds |
| Recording | 50 FPS requested, nonempty MP4 reported | Code and output; actual encoding must be checked with ffprobe |
| `VIDEO_RES` | `(960, 640)` declared | The notebook does not pass this variable to the shown camera constructor; actual resolution is unverified |

These are facts about the supplied artifact, not claims of a new remote test. Authoring this file did not connect to the remote host. Exact wheel availability, upstream tag accessibility, driver version, full dependency closure, and current instance compatibility must be checked during execution. Never replace recorded versions with a guessed modern combination when a working installation can be reused.

### 3.2 Required reuse of the notebook implementation

The notebook uses a separate Python subprocess for Genesis training and evaluation, avoiding repeated `gs.init()` in the notebook process. Preserve this process boundary. It obtains `hover_env.py`, `hover_train.py`, and `hover_eval.py` from:

`https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/v1.4.3/examples/drone/`

The training invocation is equivalent to this command, issued from the actual discovered working directory using its interpreter:

```bash
python -u hover_train.py -e UNIQUE_EXPERIMENT_NAME -b 256 --max-iterations 301
```

The notebook warns that the upstream training script deletes existing logs for the same experiment name. Therefore, never rerun it with an existing name. Back up originals, use a new run directory/name, and reject a collision before invoking upstream code. Select checkpoints by the numeric suffix in `model_<integer>.pt`, not lexical ordering, and freeze an explicit checkpoint path/hash for evaluation.

Its recording adaptation copies `hover_env.py` to `hover_env_record.py` and replaces exactly one `GUI=True,` with `GUI=False,`. Retain the original source untouched. If the exact-match assertion fails, inspect the source and report the difference instead of applying a broad replacement.

Its evaluator loads the operator-owned `cfgs.pkl` containing `(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)`, sets `reward_cfg["reward_scales"] = {}`, enables `visualize_target` and `visualize_camera`, and constructs `HoverEnv(num_envs=1, ..., show_viewer=False)`. Only deserialize operator-owned, provenance-checked pickle/model artifacts. Preserve the trained observation/action architecture and normalization.

The demonstrated inference contract is:

```python
gs.init(backend=gs.gpu, precision="32", logging_level="warning")
runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
runner.load(str(checkpoint_path))
policy = runner.get_inference_policy(device=gs.device)
obs_dict = env.reset()
with torch.no_grad():
    actions = policy(obs_dict)
    obs_dict, rewards, dones, infos = env.step(actions)
```

This is the notebook's flight contract, separate from Gymnasium's routing contract. Do not unpack the flight reset as `(obs, info)` or replace the observation dictionary with a guessed tensor.

For the demonstrated camera API, `inspect.signature(cam.start_recording)` exposes `save_to_filename` and `fps`; recording begins with `cam.start_recording(save_to_filename=str(video_path), fps=50)` and ends with `cam.stop_recording()` in `finally`. The notebook also contains a legacy branch where saving arguments are passed to `stop_recording` and frames are manually rendered. Preserve signature checks and use only the branch verified for the installed source. Verify frame cadence, actual resolution, and video duration; successful physics does not establish successful recording.

The notebook evaluator allows automatic environment resets. Its saved video is therefore not proof that a patrol reached all intended goals. The patrol implementation must observe termination/reset before it can produce a false arrival from a reset pose.

### 3.3 Feasibility and selected approach

Use a hierarchical design: CPU analytical MaskablePPO for routing and the existing Genesis/RSL-RL PPO for motor control. This preserves the demonstrated GPU workflow and keeps routing experiments cheap. Two alternatives are deliberately outside the selected design: training everything jointly in Genesis increases debugging/training cost, and a heuristic-only router fails the learned-routing project requirement. Fixed, greedy, and exact routing remain comparison baselines.

The three-checkpoint routing problem is small and tractable. It permits an exact static reference through six permutations. The principal feasibility uncertainty is flight reliability under external target changes: stale observations, automatic target resampling, hidden resets, target distribution limits, and dwell stability can all invalidate a patrol even with a good one-goal video.

Reuse a compatible checkpoint first. Validate one external goal, then two successive goals, then complete patrols. Retrain or fine-tune flight only when these measured checks justify it. Keep all scenario locations in the measured controller envelope. A tight 8-hour GPU allowance is realistic as an execution cap, but neither convergence nor complete patrol success is guaranteed; record actual timings and limitations.

Additional improvements required by this specification:

- One authoritative requirement set, one CLI family, one milestone sequence, and explicit completion gates.
- Separate recorded notebook evidence, newly executed evidence, and unverified assumptions.
- Preserve the proven flight environment; isolate routing dependencies if they require incompatible PyTorch/NumPy versions.
- Define analytical cost, event timing, physical arrival, video timing, and failure accounting consistently.
- Treat authentic A/B/C patrol videos and three-checkpoint completion as mandatory.
- Deliver a local commit after remote synchronization and local verification, not just a remote commit.

## 4. SSH execution and local delivery

### 4.1 Operator-supplied runtime information

At execution time obtain only information that is missing:

- Authorized SSH alias, or host, forwarded port, username, and key/agent reference.
- Approved remote project directory and local destination checkout.
- Existing baseline directory, interpreter/environment, checkpoint/configuration, and video paths, if known.
- Whether the machine is a reused baseline instance or a fresh rental.

The SSH method is provided by the operator; the agent executes actual commands over SSH. Connection configuration and local private keys stay outside the repository. Never ask for private-key contents in a prompt. Use the instance dashboard's actual connection information, rather than assuming a username or port.

Example operator-side SSH configuration, with placeholders replaced privately:

```sshconfig
Host vast-drone
    HostName <ACTUAL_SSH_HOST>
    Port <ACTUAL_FORWARDED_PORT>
    User <ACTUAL_SSH_USER>
    IdentityFile ~/.ssh/vast_drone_key
    IdentitiesOnly yes
    StrictHostKeyChecking yes
```

Verify the host fingerprint through trusted operator information before accepting a new host. Then test from the same shell/runtime used by the coding agent:

```bash
ssh -o BatchMode=yes vast-drone 'hostname; whoami; pwd; nvidia-smi'
```

Confirm the actual GPU model and approximately 16 GB VRAM before GPU work. Record commands and results without committing private connection metadata. If the agent's shell/network/key access is blocked, explain the actual restriction and use the smallest supported way to run the agent with remote access. Do not claim remote work occurred before a successful command execution.

### 4.2 Remote command discipline

Inspect baseline files and the working interpreter read-only before any package installation. Search approved project locations first; use the historical paths from section 3 only as hints. Reuse the working interpreter or create a separate experimental environment without altering it. Do not change system CUDA or drivers as routine setup.

Transfer only reviewed source/configuration files into a dedicated project directory. Implement a documented allowlist transfer script; exclude `.ssh`, secrets, private runtime files, caches, virtual environments, large reference media, and local Git internals. Do not use destructive synchronization options or overwrite an existing project/baseline directory.

Long GPU jobs must run in a persistent `tmux` session with an explicit project script, unique run directory, log file, run ID, PID/session reference, wall-time limit, and final status file. Use an uploaded script rather than fragile nested shell quoting. The script must record actual exit status even on failure, with `RUNNING`, `SUCCEEDED`, `FAILED`, or `TIMED_OUT` states. A detached session or existing checkpoint is not proof of job success. Check process state, exit code, and required artifacts before advancing a gate.

Keep training and rendering separate, rendering only one environment. Preserve failed-job logs. Recover from disconnects by inspecting the recorded session/job before launching a replacement; never duplicate an active job. Stop only project-owned failed/timed-out processes. Do not terminate unrelated processes or the rented instance.

### 4.3 Git ownership, synchronization, and the final local commit

The authoritative deliverable is the operator's local checkout. Before changes, inspect `git status`, branch, repository root, and existing changes. If the approved destination is not a Git repository, initialize it there; do not create a repository in an unrelated parent directory. Preserve existing files and unrelated edits. Use repository-local identity only if supplied/available; ask for a missing commit identity instead of inventing it or changing global Git configuration.

Remote Git commits may be used as recovery checkpoints, but do not satisfy the local-commit requirement. Synchronize an allowlisted set of source, tests, configurations, locks, documentation, and compact evidence back to the local checkout. Copy ignored model/video artifacts separately to local `outputs/` and verify SHA-256 and sizes against the remote manifest. Do not transfer remote `.git` over local `.git`. Compare transferred source hashes and record remote revision when applicable.

Run local CPU validation in the documented local routing environment after transfer. Platform-specific GPU tests remain remote, with their evidence copied locally. Do not install Linux CUDA wheel locks on macOS; maintain platform-appropriate routing locks and a separate Linux flight lock. Validate the local file manifest and documentation, inspect the staged diff, and stage only task-owned files.

Make a local commit once the required validation passes, for example:

```bash
git diff --check
git diff --cached --stat
git commit -m "Implement adaptive three-checkpoint drone patrol"
git log -1 --format='%H %s'
git status --short
```

Stage the explicit reviewed file list before these commands. Never use `git add .` to absorb unrelated work. Record the full local commit hash in the final response and in an ignored delivery receipt, avoiding a self-referential commit hash in a tracked file.

If a required integration gate fails or the budget/access is exhausted, still preserve and locally commit the tested partial implementation when possible. Use an explicitly incomplete commit message, such as `Save partial drone patrol implementation and verified blockers`. The commit records progress; it does not convert a failed project into a completed one. Do not push or publish.

## 5. Environment discovery, freezing, and preservation

M0 is a replay of the proven workflow, not a blind reinstall. Record OS, GPU/driver, memory, interpreter path, Python version, Genesis package/source revision, RSL-RL, Torch version and `torch.version.cuda`, `torch.cuda.is_available()`, relevant asset paths, renderer/ffmpeg versions, and full package snapshot from the actual working interpreter. Redact credentials from package URLs before recording or committing a snapshot.

Capture checksums of notebook-derived sources, `cfgs.pkl`, checkpoint, and prior video. Copy sources/configuration/checkpoint to a separate immutable baseline backup. An installed package version and a requested source tag are distinct provenance facts: record both and resolve an exact commit/hash if possible.

Reproduce the baseline inference and 3-5 second camera recording in a fresh subprocess using the original source and checkpoint. Run the same target behavior before changing it. Validate a nonempty file, decode with ffmpeg, inspect frame samples, and use ffprobe for resolution, frame rate, duration, and codec. Record `PASS`, `FAIL`, or `BLOCKED`; historical user evidence remains historical if reproduction fails.

On a fresh machine with no checkpoint, reconstruct the notebook workflow in a new project/environment and train under a unique experiment name. Obtain exact recorded dependencies from available verified wheels/source artifacts. If a recorded version or source reference cannot be obtained, request the working environment snapshot/source from the operator and continue CPU work. Do not silently select a different Genesis release. Any unavoidable replacement must be explicitly justified and revalidated as a new baseline.

Produce a complete flight lock from the successful Linux environment, including the Torch wheel source/build suffix. Notebook version strings alone do not establish an installable dependency closure. Check dependency consistency and document provenance. Add CPU routing dependencies in a separate virtual environment if resolving them would mutate the flight baseline. Pin the tested `stable-baselines3`, `sb3-contrib`, `gymnasium`, plotting, configuration, and testing packages; these are additions, not notebook-proven versions. Verify their documented compatibility before locking them.

For separate environments, run a persistent CPU planner subprocess initialized once with the routing model. Use request/response JSON over standard streams, with logs on stderr. Include request ID, observation-schema ID, feature vector, mask, policy hash, action, and status; validate reply ID/shape/eligibility and enforce a timeout. The flight process exchanges only checkpoint decisions at boundaries. Do not reload the planner for every physics step or invent decisions when IPC fails. A single environment is acceptable only after a reversible, tested compatibility check establishes that it preserves the baseline.

Use measured resource limits. Begin GPU training smoke at 32 environments for two iterations. Increase to the notebook-proven 256 environments only after measuring headroom; keep at least 2 GB free VRAM where feasible. The observed 256-environment baseline is useful evidence, not permission to hardcode 2,048 environments. On OOM, preserve logs and retry with half as many environments in a new process. Do not keep increasing batches or train indefinitely. Check disk space before checkpoints, frame capture, and final video.

## 6. Architecture and repository structure

The upper policy learns in an analytical Gymnasium environment without Genesis physics. The lower policy learns or reuses training in Genesis with RSL-RL PPO. Both execute in the final mission. Do not describe them as joint or end-to-end training.

Use small components with explicit responsibilities:

| Component | Contract |
| --- | --- |
| Scenario and objective | Validated seeded scenario, snapshots, shared analytical cost and event logic |
| Routing environment/agent | Masked discrete choices, train/load/infer, no Genesis import |
| Baselines and evaluation | Fixed/greedy/exact-static comparisons on common scenario inputs |
| Flight environment/agent | Notebook-derived motor control, checkpoint contracts, pre-reset state capture |
| Target adapter | Set an external world-frame goal and refresh all target-dependent state before inference |
| Patrol runner | Arrival-based state machine, scenario events, failure detection, logging |
| Recorder and compositor | Actual Genesis frames, frame timestamps, deterministic offline HUD/editing |
| Remote/delivery tools | Durable jobs, budget ledger, manifest, reviewed transfer, local verification |

Suggested layout; create only modules that have a concrete purpose:

```text
spec.md
README.md
pyproject.toml
.gitignore
LICENSE
THIRD_PARTY_NOTICES.md
requirements/
  flight-linux.lock.txt
  routing-linux.lock.txt
  routing-local.lock.txt
configs/
  routing_train.yaml
  flight_train.yaml
  flight_eval.yaml
  execution.yaml
  video.yaml
  demo_scenarios/{normal,emergency,cost_shift}.yaml
src/adaptive_drone_patrol/
  scenario.py
  objective.py
  routing_env.py
  routing_agent.py
  baselines.py
  benchmark.py
  flight_env.py
  flight_agent.py
  target_adapter.py
  patrol_runner.py
  recorder.py
  video.py
scripts/
  doctor.py
  reproduce_baseline.py
  train_routing.py
  eval_routing.py
  train_flight.py
  eval_flight.py
  run_demo_scenarios.py
  replay_decisions.py
  build_video.py
  verify_delivery.py
  run_remote_job.sh
tests/
docs/
artifacts/manifest.json
outputs/
```

If reference notebook/images are present, preserve them as inputs. Record their hashes and arrange a documented local reference location; do not include an embedded megabyte-scale notebook video in Git merely to reproduce the implementation. Commit runnable extracted/adapted source and required notices. Keep planning imports usable on CPU without Genesis, CUDA, or the flight environment. Include `--help`, validated arguments, unique outputs, actionable errors, and documented commands for every executable.

## 7. Analytical routing environment and objective

### 7.1 State, action, observation, and transitions

Use exactly three checkpoints per trained model and a base S. The action is `Discrete(3)` selecting A, B, or C. An `action_masks()` boolean vector is true exactly for unvisited checkpoints. Do not query the policy after terminal completion, when every action would be masked.

A scenario contains ID, split, seed, 3D checkpoint/base positions, priorities in `[1,5]`, directed edge multipliers in `[1,5]`, per-checkpoint service duration, objective constants, nominal speed, return-to-base flag, and exogenous event rules. Require finite values, positive nominal speed, unique node IDs, and distinct positions. Physical demo checkpoint centers must be separated by at least `2 * arrival_radius + 0.10 m`, and the initial pose must lie outside every checkpoint's arrival radius, so remaining still cannot satisfy different inspections. Default multipliers are one; the diagonal is unused. Planning geometry uses the same world coordinates as the physical mission.

The observation is a documented fixed-order `float32` vector containing current-node one-hot including S, visited flags, normalized coordinates/distances, currently observed priorities and edge costs, analytical elapsed time, and number of completed inspections. Define field offsets and fixed normalization constants in a schema saved with the model. Do not normalize using held-out data or clip extreme costs invisibly. Exclude future event time, target, and magnitude from the policy observation.

One valid routing action analytically completes a leg and its service. The routing environment returns `(observation, reward, terminated, truncated, info)`; `reset(seed=...)` returns `(observation, info)` and uses Gymnasium's seeded RNG. Full completion terminates; a configured external step limit truncates. Use an undiscounted episodic objective (`gamma=1.0`), since the benchmark sums all three leg costs.

Define invalid-action behavior explicitly: selecting an already visited checkpoint leaves visited/current/time/cost state unchanged, increments an invalid-action count, and returns zero reward with an explanatory info flag. Truncate after three invalid selections to avoid an endless loop. It never creates a second visit. Training/inference masks should make this unreachable; treat any such model output in the patrol runner or benchmark as an implementation failure. This checker-safe behavior is needed because environment checkers sample actions without respecting masks. Account for zero invalid-action rewards in the reward/cost invariant.

### 7.2 Shared cost and reward

Use one objective implementation for the environment, all baselines, replay accounting, and the static oracle. For a valid departure from i to j, snapshot the currently observed priority and multiplier and define:

```python
distance = euclidean(coords[i], coords[j])
travel_cost = distance * edge_multiplier[i, j]
travel_time = distance / nominal_speed_mps
completion_time = elapsed_model_time + travel_time + service_duration_s[j]
completion_penalty = priority[j] * completion_time
incremental_cost = alpha * travel_cost + beta * completion_penalty
reward = -incremental_cost
```

Default `alpha=1.0`, `beta=1.0`, `nominal_speed_mps=1.0`, and `service_duration_s=0.5` per checkpoint. The two coefficients scale abstract units; the sum is not energy or physical time. Logical multipliers affect travel cost, not actual flight speed or nominal time. Store the constants with every scenario and model.

Set `return_to_base=false` for required demos. Optional return tests must add the last-to-base travel cost once, update modeled time once, and add no priority penalty for S. It must not add another checkpoint action or double-count termination costs. Require `total_cost == -sum(rewards)` within numerical tolerance for all accounted transitions.

Priorities multiply completion times so order matters. Do not reward the sum of inspected priorities independently of order. A hand-check fixture with service zero, S=(0,0), A=(1,0), B=(0,1), priorities A=1/B=5, unit multipliers/speed, and alpha=beta=1 gives costs `7 + 6*sqrt(2)` for A then B and `7 + 2*sqrt(2)` for B then A. Use this as an objective unit test; it is mathematical test data, not an experimental result.

At physical execution, assess the modeled leg only after the goal and service are actually completed, using the departure snapshot. Failed legs create no completed inspection; preserve partial modeled cost and report physical failure separately. Record measured flight duration/distance in different fields. Do not feed measured duration into a model trained on nominal analytical duration without a documented retraining change.

### 7.3 Training and event semantics

Train a real `sb3_contrib.MaskablePPO` with `MlpPolicy`, CPU device, fixed masks, and a small MLP. Suggested starting configuration: two 64-unit hidden layers, learning rate `3e-4`, rollout length 256, batch size 64, ten PPO epochs, and `gamma=1.0`. Save exact hyperparameters, seed, versions, normalization schema, training logs, and `.zip` checkpoint. Validate package-specific configuration support rather than inventing arguments.

Use three training seeds, 42/43/44. Begin with a static curriculum, then train/fine-tune on an event-inclusive mixture: 50% static episodes, 25% priority-update episodes, 25% cost-update episodes. Events occur after the first completed inspection and before the next action. Sample new event values from the configured distributions at that boundary; make that process depend only on currently observed state and completed-inspection count. Demonstration events may be fixed, but future event details must remain hidden until delivery.

Default budget per seed: 50,000 static plus 50,000 event-inclusive steps; extend to at most 200,000 total per seed if validation identifies a useful improvement. Account for PPO rollout rounding. Select the single final routing checkpoint by validation metrics including invalid-action absence and event-response behavior; freeze it before held-out testing. A static-only model does not satisfy dynamic-training acceptance.

Use disjoint documented scenario split seeds and manifests: training below 10,000; validation 10,000-19,999; held-out static 20,000-20,099; held-out dynamic 30,000-30,049; demonstration selection 40,000-40,099. Training initialization seeds are distinct from scenario seeds. Save the generated scenario files so every planner sees identical inputs; do not rely only on a seed to hide generator changes.

Use mask-aware evaluation/callbacks. At deterministic inference pass the current mask explicitly to `model.predict`. Prefer a simple vectorization setup initially. If using subprocess vectorization, implement `action_masks()` inside the environment. Revalidate against the locked package version.

### 7.4 Baselines and reported metrics

Compare fixed A-B-C, greedy minimizing current `distance*multiplier/priority`, learned routing, and exact static enumeration. Break ties by checkpoint ID. All planners use the same objective and base-return setting. Exact enumerates all six static tours and cannot exceed any valid static tour's cost within tolerance.

Use 100 held-out static and 50 held-out dynamic scenarios for each of the three routing training seeds. Report per-seed results and across-seed mean/std; do not treat correlated evaluations of the same scenarios as independent samples. Dynamic comparisons use the same exogenous event definition for every planner: for a fixed checkpoint update, skip and log the update if that checkpoint has already been visited. Do not choose a different emergency target independently for each planner. There is no required dynamic clairvoyant oracle.

Report analytical total cost, priority-weighted completion penalty, nominal distance, completion ratio, paired differences versus fixed/greedy, static RL/exact ratio, invalid-action count, and dynamic decision-change rate. Define that rate over cases with at least two eligible choices and compare the actual post-event inference to a no-event inference on an identical copied boundary state. Static exact is a reference, not a required RL performance target. Report underperformance honestly.

Save scenario-level CSV/JSON and plots for cost comparison, scenario orders, event response, and actual physical trajectory. Label model units separately from seconds/meters from physical telemetry. Require nonempty measured data and disclose sample sizes. Test-set results must not be used to choose a new model or retune a scenario.

## 8. Flight policy, target injection, and physical acceptance

### 8.1 Inspect the actual source before adapting it

Start from the notebook-derived source and environment, not a tutorial for a different release. Document the exact signatures and tensors for `HoverEnv.reset`, `step`, `get_observations` if present, the target/command representation, reward-relative state, automatic goal resampling, termination/reset, motor mapping, and camera methods. Record observation offsets, shape/type, quaternion order, world/body frames, simulator/environment origins, timestep, and command bounds. Verify that the checkpoint architecture and configuration match these values.

Do not assume a method to disable resampling or resets exists. Implement a small, attributed adaptation of the known source where necessary. Preserve reward/control conventions and test every modified boundary. Capture pose, velocity, termination cause, and reset status before upstream reset can overwrite them; alternatively disable evaluation auto-reset explicitly in the adapted step path. A reset or fall must fail the current mission, even if the reset state is near the goal.

### 8.2 Target adapter contract

Implement a public project adapter with these logical operations; these are required project interfaces, not assertions that Genesis already provides them:

```python
adapter.set_target(world_xyz)
observation = adapter.current_observation()
step_result = adapter.step(actions)
```

`set_target` validates shape, bounds, device, finite values, and coordinate frame. It updates the task command/goal marker and every target-dependent observation/reward/history buffer required by the inspected source. The next inference must consume the updated relative target. Preserve previous motor actions and physical state unless the source contract requires a documented change; never reset the drone simply to switch goals.

During patrol evaluation, prevent automatic goal resampling from overriding the selected checkpoint. Capture the commanded target, effective task target, and target-relative observation in logs. Include an integration assertion that effective target and selected waypoint agree before each action. A target mismatch is a failed mission, not permission to substitute a nearby goal.

Initialization may use the baseline's normal reset/takeoff behavior once at mission start. No waypoint transition may teleport position, call a hidden reset, or replace physics with a drawn trajectory. Record any preflight stabilization time separately. Set base/waypoint coordinates relative to the verified stable operating region and keep the same positions in all three demo scenarios.

### 8.3 Arrival, dwell, and failure rules

Use actual three-dimensional Genesis pose and velocity. Default arrival radius is 0.20 m and maximum arrival speed is 0.30 m/s. Both must hold continuously for 0.20 simulation seconds; compute the required consecutive steps from the verified `env.dt`. An additional inspection dwell completes a total of 0.50 seconds inside those position/speed limits, including the qualifying arrival hold. Reset the dwell timer on any violation. Record `ARRIVAL_CONFIRMED` after the first hold and `WAYPOINT_REACHED` only after service completion.

Choose flight-safe source-derived altitude and tilt limits from the baseline configuration and document the numbers before evaluation. Do not guess a quaternion convention. A floor violation, excessive tilt/deviation, nonfinite state/action, reset/done, inconsistent goal, or loss of controller/planner communication fails the mission. Defaults: maximum 30 simulation seconds per leg, 120 seconds per mission, and a separately documented preflight limit. Physical time and monotonic wall time have separate watchdogs so a stalled simulator cannot run indefinitely.

Freeze acceptance tolerances before final trials. Any change must be explained, versioned with a configuration hash, and retested; do not relax tolerances after seeing a failure merely to declare success. Never mark a fly-through, an approach, a reset pose, or an intended target as inspected.

### 8.4 Progression and conditional additional training

1. Reproduce the notebook's one-goal checkpoint inference and authentic recording.
2. Command one external nearby goal and verify observation/goal consistency and physical arrival.
3. Execute two successive external goals with the same checkpoint and no intermediate reset; log actual target switch and arrival results.
4. Evaluate the three-checkpoint geometry and arrival/dwell rules using short sequential trials.
5. Freeze the final flight checkpoint/configuration and execute A/B/C missions.

Start with nearby displacements, approximately 0.3-0.6 m as a trial configuration only if compatible with the inspected source; measure the operating envelope before declaring them reachable. Include at least three seeded sequential-goal trials and report every attempt. Reuse the same validated geometry across A/B/C. Narrow geometry on development/validation trials before freeze if necessary, while preserving distinct targets and observable flight.

If target-conditioned reuse fails, first distinguish adapter errors from controller generalization problems. Only then fine-tune/retrain in a new run. Preserve the 17-input/4-output architecture if reusing the old weights; an observation change requires a new compatible model, not a silently misloaded checkpoint. Use training targets matching the eventual distances, altitude, switching pattern, and dwell needs. Begin with the upstream reward and document any changes. The final flight checkpoint must be genuinely trained PPO, even when extra training is needed.

Two-iteration smoke training verifies execution/checkpoint plumbing, not convergence or task competence. Long training begins only after that smoke, save/load, memory, and rendering checks. Track budget, finite losses/rewards, saved iterations, validation arrivals, failures, and the reason for each retry. Preserve the old known-good model. Do not use a scripted or classical motor controller in the final patrol while claiming a learned flight policy.

## 9. Patrol execution and the three scenario demonstrations

### 9.1 State machine and event order

Implement this logical state machine:

```text
INIT -> SELECT_TARGET -> FLY -> ARRIVAL_HOLD -> INSPECT -> UPDATE_SCENARIO
          ^                                                   |
          +---------------------------------------------------+
all three inspected -> DONE
optional return enabled -> RETURN_BASE -> DONE
any safety, reset, timeout, model, or communication failure -> FAILED
```

A hold violation returns to FLY without advancing the route. After physical inspection completion, update visited/current-node state and modeled cost from the departure snapshot. Apply the external event scheduled for that boundary, snapshot the new state, obtain a fresh masked routing inference, set the flight target, refresh observations, and begin the next leg. This sequence must be identical in the analytical routing transition and the physical runner's logical state update.

The routing model never advances the physical mission merely by choosing an action. The physical runner owns arrivals. For split processes, maintain logical routing state in the runner/shared pure objective code and request inference from the CPU planner; do not let the planner's training `step()` instantly mark a physical leg complete.

If optional base return is enabled, select S deterministically after all three inspections and verify its physical arrival. S is not a routing action, adds no visited checkpoint, and cannot conceal a failed patrol. Required demonstrations leave this option disabled.

### 9.2 Scenario definitions

Use one frozen routing model, one frozen flight model, the same three coordinates, same starting state, same dwell/tolerances, and reproducible seeds for all scenarios. The initial priorities and multipliers for the paired normal/emergency/cost-shift demonstrations must match so their first decision is directly comparable.

- **Normal:** no updates, nominal edge multipliers, configured positive priorities. Record the complete learned visit order and physical execution.
- **Emergency:** after the first actual completed inspection, raise a specified still-unvisited checkpoint's priority, for example from 1 to 5. Apply the event before the next inference. Record old/new values and the real selected action.
- **Cost shift:** after the first actual completed inspection, increase specified outgoing logical edge multipliers. Apply the event before the next inference. Explain that these are abstract route penalties; the drone does not perceive obstacles and its motor-control target remains a waypoint.

Select the emergency checkpoint and affected edges during demonstration-set search so they are eligible at the first boundary. Freeze the selected config before final recording. Do not choose them after inspecting final video. Selection is a disclosed illustrative-case search, separate from held-out benchmark claims.

Require evidence of adaptive decisions for **both event types**: on the same copied pre-event boundary state, run the frozen model with the actual updated state and with a no-event state, using deterministic inference and the same mask. Log both policy observations and actions. At least one emergency demonstration and one cost-shift demonstration must show a changed selected target. The counterfactual is planner-only evidence; label it as such and never present it as another physical trajectory.

If either event produces no policy change, log `NO_POLICY_CHANGE`; inspect event magnitude, representation, and validation performance within the authorized budget. Any further training produces a new frozen model and requires a new final evaluation. Do not write the desired action into the policy output or fall back to a heuristic without disclosure. A truthful failure report remains an unmet adaptation gate.

### 9.3 Physical demonstration acceptance

For each scenario, perform three recorded-or-telemetry-backed seeded attempts after freezing the policy pair. Report all attempts and the actual success ratio; these nine runs are a small demonstration sample, not a statistical reliability guarantee. The same initial conditions are used across scenario types for each seed.

At least one attempt per scenario must inspect A, B, and C exactly once, finish with no reset/fall/timeout, and produce a decodable raw Genesis video with matching events/telemetry. Use the paired seed with complete comparable A/B/C runs for the final narrative if available; disclose how representative clips were chosen and retain failed attempts. If no such complete attempt exists for a scenario, final project status is incomplete.

Record a single execution and its camera frames together. A telemetry file cannot reconstruct missing Genesis images. If a new simulation replay is recorded instead, it is a new run with its own telemetry, hashes, and outcomes, and must be evaluated again. Do not join old telemetry to new video solely because the seeds match; GPU physics need not be bitwise reproducible.

## 10. Evidence schemas, replay, and metrics

Use unique run directories and append-only decision/event logs. Each run contains a schema version, run/scenario ID, split/seed, policy hashes, configuration hashes, source revision, environment reference, timestamps, and exit status. Events use strictly increasing event sequence IDs; several events may share a simulator timestamp.

`decisions.jsonl` must record the following for each decision:

- Event sequence, decision index, simulation step, `sim_time_s`, and separate `model_time_s`.
- Current node, visited/eligible checkpoint IDs, actual checkpoint action index and target.
- Complete feature vector, observation schema, current mask, priorities, cost snapshot or its hash/path.
- Actual selected action, policy checkpoint hash, inference mode, and event source `trained_policy`.
- Counterfactual record ID when applicable; do not confuse actual and hypothetical actions.

Also emit `SCENARIO_START`, `TARGET_SELECTED`, `LEG_STARTED`, `ARRIVAL_CONFIRMED`, `WAYPOINT_REACHED`, `PRIORITY_UPDATED`, `COST_UPDATED`, `RESET_DETECTED`, `LEG_FAILED`, `NO_POLICY_CHANGE`, and terminal `MISSION_FINISHED`/`MISSION_FAILED` as applicable. Store pre/post event state and trigger boundary. Record completion only after physical service. Include explicit termination cause and completed/intended routes in the summary.

`telemetry.csv` must include run ID, simulation step/time, monotonic wall elapsed time, actual xyz pose, quaternion with declared convention, linear speed/velocity, effective target xyz/ID, distance to goal, controller action, arrival/dwell progress, runner state, reset/done/safety flags, and frame reference when emitted. Copy tensors into durable snapshots; do not log mutable references that later change.

`frames.csv` maps camera frame index/PTS to the actual simulator step/time at capture. Verify the relationship for the selected recording API. Do not infer it solely from requested FPS, host wall time, or the declared `VIDEO_RES` variable. If automatic recording cannot expose reliable timing, inspect and instrument the pinned source or use a verified explicit-frame capture path; validate it in a camera smoke before producing final overlays.

`mission_summary.json` includes actual completion status, per-leg departure/arrival/service times, arrival residuals, actual path length from consecutive non-reset samples, measured duration, analytical cost components, selected/visited order, event/adaptation outcomes, crash/reset/timeout counts, artifacts, and policies/configurations. Keep all failed attempts in the evaluation table; do not compute success rates only from selected videos.

A deterministic offline replay tool reconstructs tactical maps, logged decisions, and HUD state from the event stream and telemetry. It does not recreate missing physics footage. Validate that every shown visit has a service event, every actual trajectory point comes from telemetry, all IDs/references resolve, and event ordering is consistent. Physical distance/time and nominal planning metrics must have distinct column names and chart labels.

## 11. Video, charts, report, and originality

### 11.1 Authentic recording and editing

First preserve a working 3-5 second baseline recording. Final raw footage must come from the Genesis renderer executing the trained policy pair. Keep the drone/markers visible using the proven camera initially; change camera settings only in a new smoke-verified adaptation. Render a single environment. Preserve raw clips, frame timestamps, and matching run logs.

Final output: H.264 MP4 with a widescreen canvas of at least 1280x720 and 25 or 30 FPS. The proven raw resolution/FPS may differ; resize/letterbox offline without claiming the raw footage was natively 720p. Verify encoding and duration with ffprobe and decode the whole video to detect corrupt frames. Playback acceleration, slow playback, cuts, omitted intervals, and failed legs must be explicitly identified. Prefer at most 2x acceleration unless the operator requests another rate.

Keep actual Genesis flight as the main visual for more than half the final duration. Show all three scenario patrols, with concise cuts if needed. Add a tactical map from telemetry, current target, visited count, priorities, logical costs, and observed event banners. Planned/counterfactual routes and actual trajectories must use different styles and labels. Do not display a full future route as if the online policy has already committed to it; any future rollout is labeled hypothetical.

Display clear Japanese explanations of learned next-checkpoint selection, learned four-motor control, logical costs, observed arrival, and limitations. All displayed numbers and status changes come from logs. Policy hashes/identifiers and source credits appear in metadata, credits, or accompanying documentation.

Implement deterministic offline composition with ffmpeg and a minimal plotting/image toolkit. Use a licensed installed Japanese font with documented fallback; do not commit a restricted font. Write `timeline.json` segments with source run/clip ID, source frame/time range, destination presentation range, playback rate, and cut/disclosure labels. The HUD uses the segment mapping and capture timestamps; its events must agree with the corresponding camera frames within one output-frame interval where a boundary is represented. Do not synchronize by assuming wall time equals simulation time.

Suggested 120-second narrative, adjusted to actual footage:

| Approximate interval | Content |
| --- | --- |
| 0-10 s | Inspection goal, three checkpoints, two learned policies |
| 10-35 s | Normal patrol with logged goals and actual arrivals |
| 35-60 s | Emergency priority event, fresh choice, continued physical patrol |
| 60-85 s | Logical cost update, fresh choice, continued physical patrol |
| 85-110 s | Remaining authentic flight, held-out comparison inset, actual trajectory/results |
| 110-120 s | Original contribution, limitations, source credit |

A complete raw mission can be condensed with labeled cuts; the retained full raw recording/logs establish completion. Do not replace most of the presentation with charts or pad a one-goal clip to imply a three-goal patrol. Final duration must be 60-180 seconds regardless of storyboard adjustments.

### 11.2 Charts and short Japanese report

Generate readable plots from actual benchmark/telemetry data: `cost_comparison.png`, `scenario_order_comparison.png`, `event_replanning.png`, and `real_trajectory.png`. Include units, sample counts, split/model identifiers, and limitations. Avoid unsupported claims of fastest/safest flight, battery savings, obstacle avoidance, perception, or superiority to exact enumeration.

Write a concise Japanese report, approximately 2-4 pages, covering purpose/application, architecture, observation/action/reward contracts, inherited baseline, original routing/event/target-adapter work, measured comparisons, physical outcomes, limitations, reproduction, and citations. Export a PDF with embedded/available Japanese fonts, render every page, inspect glyphs/overlaps/clipping, and repair layout before delivery. Report factual failures and modeling gaps; no invented metrics.

Credit exact Genesis example sources/assets and the operator-provided notebook. Describe the original contribution as learned inspection ordering under urgency/cost changes, verified external target control, arrival-boundary integration, and evidence-synchronized visualization. The checkpoints represent inspection locations; no sensor measurement or device inspection is simulated beyond the declared dwell unless separately implemented and verified.

## 12. Command-line interface and reproducibility

Implement the following CLI family once, using consistent option names. These are target interfaces to implement, not claims that scripts already exist. Commands run from the project root in the appropriate environment. `execution.yaml` is a private, ignored runtime file generated from operator inputs and discovery; commit `execution.example.yaml` with the schema and no credentials. Keep `flight_train.yaml` and `flight_eval.yaml` nonsecret, with portable paths or references resolved from the private execution file. Every unresolved field must cause an actionable error, not silent defaults to fictional files.

```bash
# Baseline discovery and reproduction in the flight environment.
python scripts/doctor.py --config configs/execution.yaml --check flight
python scripts/reproduce_baseline.py --config configs/execution.yaml --out outputs/baseline/replay

# Local/remote CPU routing environment; no Genesis dependency.
python -m pip install -e '.[routing,dev]'
python -m pytest -m 'not gpu' -q
python scripts/train_routing.py --config configs/routing_train.yaml --seeds 42 43 44 --out outputs/routing
python scripts/eval_routing.py --models outputs/routing/models.json --split static --episodes 100 --out outputs/evaluation/static
python scripts/eval_routing.py --models outputs/routing/models.json --split dynamic --episodes 50 --out outputs/evaluation/dynamic
python scripts/run_demo_scenarios.py --config configs/execution.yaml --mode planning-only --out outputs/planning_demos

# Flight environment on the authorized remote GPU; never overwrite old runs.
python scripts/train_flight.py --config configs/flight_train.yaml --execution configs/execution.yaml --num-envs 32 --iterations 2 --out outputs/flight/smoke
python scripts/eval_flight.py --config configs/flight_eval.yaml --execution configs/execution.yaml --mode sequential-targets --out outputs/evaluation/flight
python -m pytest -m gpu -q

# Record physics, events, telemetry, and frames during the same runs.
python scripts/run_demo_scenarios.py --config configs/execution.yaml --mode genesis --record --trials 3 --out outputs/genesis
python scripts/replay_decisions.py --run outputs/genesis/selected_normal --out outputs/previews/normal
python scripts/build_video.py --config configs/video.yaml --input-root outputs/genesis --out outputs/videos/submission.mp4

# After allowlisted synchronization, verify on the operator's local machine.
python scripts/verify_delivery.py --manifest artifacts/manifest.json --require-local-artifacts
python -m pytest -m 'not gpu' -q
python -m ruff check src scripts tests
git diff --check
```

`models.json` contains all three seed models plus an explicitly selected final model and validation rationale. The CLI resolves the frozen flight checkpoint and CPU planner interpreter from the runtime file, checks hashes/schema, and never treats `latest` as a mutable final evaluation model. `selected_normal` is a created selection record/directory pointing to a real chosen run; generate it from measured outcomes, not a prewritten success.

Each command creates its destination only if unused, or resumes only a verified matching run with an explicit option. Document concrete tested examples, the real interpreter activation, checkpoint discovery, platform locks, and model schema. Generic templates remain distinguishable from executed commands. Include one end-to-end entry point or documented command sequence that produces the deliverables without manually editing observations, routes, or output files.

The private execution configuration includes approved remote/local paths, discovered flight/planner interpreters, explicit policy hashes/paths, baseline references, scenario files, budget ledger location, process deadlines, and evidence destinations. Connection metadata is handled by the operator's SSH alias. Do not bake a private host, key path, account ID, or ephemeral root directory into public code.

## 13. Verification requirements

Use test-driven development for new objective/state/adapter logic where practical: write a meaningful failing behavior test, implement the smallest working slice, and run the relevant tests. Do not write tests that merely repeat the implementation. Documentation-only changes need consistency/file checks, not product test scaffolding. Verification evidence must precede a completion claim.

Maintain `docs/verification.md` throughout execution. Every entry contains stage, local/remote execution location, environment, exact command, timestamp, exit code, status, log/evidence paths, and interpretation. Statuses are `PASS`, `FAIL`, `BLOCKED`, `SKIPPED`, and `NOT_RUN`. A required skipped/blocked/unrun test leaves its gate incomplete. Distinguish notebook-saved historical output from new execution. A shell exit code alone does not prove task success; inspect the semantic results and artifact validity.

### 13.1 Required CPU checks

- Scenario validation, seeded regeneration, split separation, common benchmark inputs, and no future-event features.
- Gymnasium environment checks, reset/step signatures, finite shape/dtype/bounds, deterministic transition fixtures.
- Mask semantics, no duplicate visits, invalid-action behavior, no terminal all-false-mask policy query.
- Hand-computed objective fixture, departure-snapshot event accounting, reward/cost equality, and single base-return accounting.
- Exact-static lower bound over all six tours, objective parity for all baselines, deterministic tie-breaking, and paired comparisons.
- Real short MaskablePPO training/save/reload, mask-aware deterministic inference, and event-inclusive training configuration. Model competence is established separately by evaluation.
- Mocked flight state-machine cases: hold violation, dwell completion, reset near a goal, fall/NaN, target mismatch, timeout, event ordering, no phantom visit, and failed return. Label these as mocked checks.
- If environments are split: planner JSON request/response validation, mismatched request ID, invalid action, process death, and timeout produce a mission failure.
- Log schema/reference validation, immutable snapshots, monotonic event order, trajectory calculation excluding reset jumps, and failure-inclusive summaries.
- Frame/timeline mapping including 2x playback and cuts; HUD state agrees with replay events and actual frame times.
- Ruff, import/compile checks for authored modules, and local delivery manifest/source integrity. Pure CPU checks must run without importing Genesis.

### 13.2 Required remote GPU checks

- Actual RTX 5060 Ti/16 GB identity, working interpreter, CUDA availability, exact source/configuration/checkpoint compatibility.
- Baseline one-goal inference and short headless camera replay, ffprobe/decode/sample-frame checks.
- Two-iteration, small-environment flight training smoke with finite outputs, checkpoint save/reload; use a fresh run name even when the final flight model is reused.
- External target update modifies the intended observation entries before inference, preserves physical state, and prevents automatic goal resampling.
- Pre-reset detection: a reset cannot create an arrival or completed inspection; real runtime termination is recorded accurately.
- One externally commanded arrival, two successive goals without reset, and seeded geometry/dwell evaluation with actual pose/velocity.
- All three scenarios run with the same frozen policy pair, all attempts retained, at least one complete physical three-checkpoint recorded patrol per scenario.
- Both event types are applied only after real service completion, and each has a logged deterministic counterfactual action-change demonstration.
- Raw videos decode and match their own telemetry/frame maps; final H.264 MP4 decodes, lasts 60-180 seconds, has Japanese captions and labeled edits/speed changes.
- Visually inspect representative frames before/after each event, each arrival, all scenario transitions, titles/charts/credits; ensure the drone is visible and labels do not overlap. Inspect all report PDF pages.

CPU mocks, analytical arrivals, a saved checkpoint, a generated model file, and an existing video do not substitute for these physical checks. Record observed memory peaks and GPU phase duration; never extrapolate proof of convergence from smoke training.

## 14. Ordered milestones and failure handling

| Stage | Work | Required exit evidence |
| --- | --- | --- |
| M0: Access and baseline | Inspect local state, verify SSH/GPU, locate/freeze baseline, reproduce one-goal inference and camera | Proven-baseline document, exact environment/source/checkpoint hashes, decodable short clip |
| M1: Analytical foundations | Scenario schema, shared objective/events, masks, baselines, state-machine interfaces | Passing CPU correctness checks, explicit costs and event semantics |
| M2: Learned routing | Three seeded training runs, validation selection, static/dynamic evaluation, A/B/C planning replay | Real checkpoints/logs, 100 static/50 dynamic cases per seed, both event-response demonstrations |
| M3: Sequential flight | Adapter, reset capture, single/two-goal checks, training smoke, conditional fine-tuning | Correct observation updates, actual arrival/dwell evidence, validated geometry/checkpoint |
| M4: Complete integrated patrols | Frozen policy pair, three attempts per scenario, same-run recording/telemetry | Complete recorded A/B/C patrols, truthful summaries, actual boundary updates and adaptation evidence |
| M5: Submission and local delivery | HUD/video, figures/report/PDF, notices/locks, artifact transfer, local tests and commit | Verified 60-180 s video, readable PDF, valid local manifest, local commit hash |

Within M0-M3, perform small camera/target checks early to expose blockers. Favor correctness and evidence over cosmetic features. Long additional flight training begins only after the baseline replay and target-adapter diagnosis. Avoid unrelated features such as obstacles, perception, battery simulation, hardware deployment, or a fourth checkpoint until the required outcome is complete; they are not part of this contract.

On failure, preserve logs and artifacts, identify the failing layer, correct only justified causes, and rerun the relevant gate. An API error is different from model generalization failure or rendering failure. Keep failed attempts and model/configuration revisions visible. Do not silently switch controller algorithms, exclude poor benchmark cases, rewrite telemetry, relax arrival rules, or replace camera footage with synthetic animation.

If the GPU budget, access, package availability, renderer, or model reliability prevents completion, finish independent CPU work, synchronize partial results and reproducible commands, locally commit tested progress, and report each unmet gate. Ask for only the missing information or additional budget needed. Do not leave the operator with a fictitious successful submission. Resume from the last verified checkpoint when the blocker is resolved.

## 15. Final acceptance and agent response

The project is complete only when every item below has evidence:

- [ ] Notebook-derived one-goal training/inference/recording contracts are preserved, reproduced, and attributed.
- [ ] Actual authorized SSH command execution and RTX 5060 Ti/16 GB runtime checks are recorded.
- [ ] Exact tested flight and routing dependencies are locked without mutating the preserved baseline.
- [ ] Analytical objective, masks, event order, and static oracle pass correctness checks.
- [ ] Three real routing training seeds include post-arrival events; validation selects one frozen final model.
- [ ] 100 static and 50 dynamic held-out scenarios per training seed have real paired benchmark data and honest figures.
- [ ] The learned flight policy is loaded with a compatible configuration and physically verifies external/sequential targets.
- [ ] Reset/fall/timeout/goal mismatch cannot create a false visit; pose/speed/dwell establish completion.
- [ ] Normal, emergency, and cost-shift scenarios each have at least one complete three-checkpoint Genesis patrol, authentic raw footage, and matching telemetry.
- [ ] Both priority and logical-cost events occur after a real inspection; each event type has genuine logged action-change evidence.
- [ ] All final patrols use the same frozen policy pair and geometry; all attempted missions remain in reported results.
- [ ] HUD/tactical map/plots derive from real logs with verified source-frame and presentation-time mapping.
- [ ] The final video decodes, lasts 60-180 seconds, prominently shows actual flight, and labels Japanese explanations, cuts, and playback changes.
- [ ] Japanese README/report/PDF, complete source, notices, configuration, locks, and compact evidence are delivered.
- [ ] Large artifacts have been copied locally with matching hashes before the ephemeral instance becomes unavailable.
- [ ] Required remote and local checks pass; no required gate is substituted by a mock or marked unrun.
- [ ] The reviewed task files are committed in the operator's local repository, and the actual full commit hash is available.
- [ ] GPU execution stayed within the authorized 8-hour cap or a separately approved extension.

The final response to the operator must be concise and factual: complete/incomplete status, implemented behavior, local commit hash and checkout path, key test results, held-out routing measurements, actual per-scenario flight completion counts, delivered video/report paths, GPU phase time, and any unmet requirements with exact recovery commands. Use Japanese for conversational explanations and English for code/comments. Do not claim publication or course submission.

## 16. Reference policy

The notebook, its operator-owned working environment, and inspected pinned source take priority over unrelated or newer example snippets. Resolve disagreement through source/runtime evidence rather than selecting a convenient API. No remote baseline execution occurred while authoring this specification. The upstream source URLs for the recorded tag could not be verified through authoring-time web access; the notebook's saved outputs are the evidence for its recorded versions.

Supporting official references:

- [Genesis source repository](https://github.com/Genesis-Embodied-AI/genesis-world). Obtain the notebook's `v1.4.3` source from the working copy or exact reference and record its hash; do not assume a web page fetch establishes compatibility.
- [MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html). It documents explicit inference masks, mask-aware evaluation, and checker/vectorization caveats; consult the locked-version documentation at implementation time.
- [Gymnasium custom environment guide](https://gymnasium.farama.org/introduction/create_custom_env/). Verify reset/step contracts and seeded environment behavior against the chosen locked release.
- [Vast.ai SSH connection guide](https://docs.vast.ai/guides/instances/connect/ssh). Use the actual instance connection information and operator-controlled authentication.

External documentation supplements this contract. It does not override the mandatory patrol outcome, the proven notebook baseline, the English-comment rule, the GPU budget, or the local-commit requirement.

## 17. Operator kickoff prompt

The operator can give the coding agent this instruction together with this file and the private runtime information:

```text
Read spec.md in full and autonomously implement it through the final local Git commit.
Use the supplied authorized SSH method to work on my already rented Vast.ai Linux
RTX 5060 Ti 16 GB instance. Reuse the demonstrated RL_drone_hovering.ipynb baseline
and preserve its working source, package versions, configuration, and checkpoint.
All three scenarios must complete all three checkpoints in real Genesis physics.
Verify both arrival-boundary event types, record actual footage and telemetry,
produce the 1-3 minute submission video and report, and run remote and local checks.
The GPU execution allowance is 8 hours including retries. Keep source comments,
docstrings, and the specification in English; write audience-facing explanations
in Japanese. Synchronize the completed artifacts to my local checkout and commit
the reviewed task files there. Do not push, publish, or submit. Ask for missing
runtime information when necessary; do not stop solely to reapprove the plan.
Report actual evidence and explicitly identify any incomplete requirement.
```

Supply the SSH alias/method, approved remote project directory, local checkout path, and available baseline interpreter/source/checkpoint paths as private execution input. These deployment-specific values are intentionally supplied at run time and are not unresolved design requirements.
