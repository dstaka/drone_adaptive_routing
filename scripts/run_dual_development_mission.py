"""One predeclared B-static development mission, preserving unsuccessful outcomes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from adaptive_drone_patrol.dual_coordination_baselines import StaticBaseline
from adaptive_drone_patrol.dual_coordination_env import DualCoordinationEnv
from adaptive_drone_patrol.dual_coordination_progress import RunJournal, atomic_json
from adaptive_drone_patrol.util import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declaration", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--journal-root", required=True)
    args = p.parse_args()
    records = json.loads(Path(args.declaration).read_text())
    cases = [row["case"] for row in records if row["case"]["case_id"] == args.case_id]
    if len(cases) != 1 or cases[0]["split"] != 15:
        raise ValueError("One exact predeclared development case required")
    journal = RunJournal(args.journal_root)
    attempt = journal.start("development-B-static", args.case_id, category="development physical pilot")
    started = time.monotonic()
    env = None
    try:
        env = DualCoordinationEnv(
            args.original, args.output, cases[0], json.loads(Path(args.design).read_text()), record_raw=False
        )
        baseline = StaticBaseline(cases[0]["starts"], cases[0]["goals"])
        actions = []
        last = time.monotonic()
        while not env.terminated:
            positions = [row["drone_xyz"] for row in env.fixture.sample()["drones"]]
            decision = baseline.select(env.mission, positions)
            tick = env.mission.tick
            _, reward, _, info = env.step(decision.action)
            actions.append(
                dict(
                    tick=tick,
                    decision=asdict(decision),
                    reward=reward,
                    terminal_reason=info["terminal_reason"],
                )
            )
            if time.monotonic() - last >= 30:
                journal.heartbeat(
                    attempt,
                    completed=sum(g.status == "DONE" for g in env.mission.goals),
                    total=10,
                    reason=f"simulated {env.mission.tick * 0.01:.2f}s; development only",
                )
                last = time.monotonic()
        env.close()
        root = Path(args.output)
        atomic_json(root / "proposed-actions.json", actions)
        result = dict(
            scope="DEVELOPMENT_ONLY",
            production_transitions=0,
            formal_acceptance=False,
            case_id=args.case_id,
            method="B-static",
            terminal_reason=env.mission.terminal_reason,
            team_time=env.mission.tick * 0.01,
            completed_counts=env.mission.completed_counts,
            final_qpos=env.fixture.data.qpos.tolist(),
            final_qvel=env.fixture.data.qvel.tolist(),
            minimum_control_boundary_clearance_m=env.ledger.minimum_clearance_m,
            floor_minima_m=env.ledger.floor_minima_m,
            safety_failures=env.ledger.failures,
            decisions=len(actions),
            elapsed_including_compile_s=time.monotonic() - started,
            compilation_s=env.compilation_wall_seconds,
            action_sha256=sha256(root / "proposed-actions.json"),
            replay_bytes=sum(x.stat().st_size for x in root.glob("replay-*")),
            replay_files=[
                dict(path=x.name, bytes=x.stat().st_size, sha256=sha256(x))
                for x in sorted(root.glob("replay-*"))
                if x.is_file()
            ],
        )
        atomic_json(root / "pilot-result.json", result)
        journal.finish(
            attempt,
            status="passed" if env.mission.terminal_reason == "success" else "failed",
            reason=env.mission.terminal_reason,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["terminal_reason"] == "success" else 1
    except BaseException as error:
        journal.finish(attempt, status="failed", reason=f"{type(error).__name__}: {error}")
        if env is not None:
            env.close()
        raise


if __name__ == "__main__":
    sys.exit(main())
