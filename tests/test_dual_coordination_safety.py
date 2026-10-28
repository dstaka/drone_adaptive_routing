"""Conservative between-state motion checks, and permanent safety failures."""

import numpy as np
import pytest


def test_swept_spheres_detect_crossing_with_clear_endpoints():
    from adaptive_drone_patrol.dual_coordination_safety import swept_sphere_clearance

    distance, fraction = swept_sphere_clearance([-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], 0.1, 0.1)
    assert distance == pytest.approx(-0.2)
    assert fraction == pytest.approx(0.5)
    distance, _ = swept_sphere_clearance([0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0], 0.1, 0.2)
    assert distance == pytest.approx(1.7)


def test_geometry_surface_support_for_rotated_box_and_disk():
    from adaptive_drone_patrol.dual_coordination_safety import vertical_extent

    theta = np.pi / 4
    r = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    assert vertical_extent("box", [0.1, 0.2, 0.3], r) == pytest.approx(0.4 / np.sqrt(2))
    assert vertical_extent("cylinder", [0.1, 0.3, 0], r) == pytest.approx(0.4 / np.sqrt(2))


def test_wash_uses_whole_other_shape_and_vertical_interval():
    from adaptive_drone_patrol.dual_coordination_safety import wash_sphere_incursion

    assert wash_sphere_incursion([0, 0, 2], [0.65, 0, 1.5], 0.10)
    assert not wash_sphere_incursion([0, 0, 2], [0.75, 0, 1.5], 0.10)
    assert not wash_sphere_incursion([0, 0, 2], [0, 0, 2.3], 0.10)
    assert wash_sphere_incursion([0, 0, 2], [0, 0, 0.95], 0.10)


def test_latched_failure_cannot_be_repaired_by_later_safe_state():
    from adaptive_drone_patrol.dual_coordination_safety import SafetyLedger

    ledger = SafetyLedger()
    ledger.observe(
        {
            "world_time": 0.01,
            "min_clearance_m": 0.19,
            "floor_minima_m": [0.8, 0.8],
            "contacts": [],
            "wash_incursions": [],
            "finite": True,
            "constraint_max_m": 0,
            "max_roll_pitch_deg": 0,
        }
    )
    ledger.observe(
        {
            "world_time": 0.02,
            "min_clearance_m": 2.0,
            "floor_minima_m": [0.8, 0.8],
            "contacts": [],
            "wash_incursions": [],
            "finite": True,
            "constraint_max_m": 0,
            "max_roll_pitch_deg": 0,
        }
    )
    assert not ledger.safe
    assert ledger.failures[0]["world_time"] == 0.01
    assert ledger.minimum_clearance_m == 0.19


def test_supervisor_qualification_is_required_and_hold_remains_obstacle():
    from adaptive_drone_patrol.dual_coordination_safety import ReferenceSupervisor
    from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference, KinematicState

    supervisor = ReferenceSupervisor()
    refs = [ContinuousReference([-2, 0, 3]), ContinuousReference([2, 0, 3])]
    states = [KinematicState([-2, 0, 3], [0, 0, 0]), KinematicState([2, 0, 3], [0, 0, 0])]
    desired = [
        r.retarget(target, 0, actual_state=s) for r, target, s in zip(refs, [[2, 0, 3], [-2, 0, 3]], states)
    ]
    states = [KinematicState(*r.sample(2)[:2]) for r in desired]
    result = supervisor.decide(desired, states, 2, priority=1)
    assert any(result.held)
    assert result.horizon_s >= 2
    assert result.qualification == "DEVELOPMENT_ONLY"
    # The immobile drone at the desired endpoint remains in the swept envelope.
    states[1] = KinematicState([2, 0, 3], [0, 0, 0])
    result = supervisor.decide([desired[0], refs[1]], states, 2, priority=1)
    assert result.held[0] or result.failure is not None


def test_batched_predictor_preserves_scalar_decisions_and_braking_trajectories():
    import math
    from adaptive_drone_patrol.dual_coordination_reference import ContinuousReference, KinematicState
    from adaptive_drone_patrol.dual_coordination_safety import ReferenceSupervisor, swept_sphere_clearance

    class ScalarReferenceSupervisor(ReferenceSupervisor):
        # Independent pre-optimization scalar oracle, including early-return
        # minimum and interval wash logic; do not call the batched predictor.
        def predict(self, references, time_s, horizon_s, radii):
            times = np.linspace(time_s,time_s+horizon_s,max(1,math.ceil(horizon_s/.1))+1)
            minimum = float('inf')
            for start,end in zip(times,times[1:]):
                a0,b0 = (r.sample(start)[0] for r in references)
                a1,b1 = (r.sample(end)[0] for r in references)
                rr = [radii[i]+r.bounds().speed*(end-start) for i,r in enumerate(references)]
                distance,_ = swept_sphere_clearance(a0,a1,b0,b1,*rr)
                minimum = min(minimum,distance)
                if distance < .20:
                    return True,minimum
                horizontal,_ = swept_sphere_clearance(np.r_[a0[:2],0],np.r_[a1[:2],0],
                    np.r_[b0[:2],0],np.r_[b1[:2],0],*rr)
                low_a,high_a = min(a0[2],a1[2])-rr[0],max(a0[2],a1[2])+rr[0]
                low_b,high_b = min(b0[2],b1[2])-rr[1],max(b0[2],b1[2])+rr[1]
                if horizontal < .60 and (low_b <= high_a and high_b >= low_a-1
                                         or low_a <= high_b and high_a >= low_b-1):
                    return True,minimum
            return False,minimum

    scenarios = [(((-3,-3,2),(3,3,2)),((1,-3,2),(-1,3,2))),
                 (((-3,0,2),(0,-3,2)),((3,0,2),(0,3,2))),
                 (((-3,0,2),(3,0,2)),((3,0,2),(-3,0,2))),
                 (((-3,0,2),(0,0,2)),((3,0,2),(0,0,2)))]
    reasons = set()
    for starts,goals in scenarios:
        refs = [ContinuousReference(p).retarget(g,0,actual_state=KinematicState(p,(0,0,0)))
                for p,g in zip(starts,goals)]
        for now in (.2,1.2,2.4):
            states = [KinematicState(r.sample(now)[0]+[.01,-.02,.005],r.sample(now)[1]) for r in refs]
            for priority in (1,2):
                actual = ReferenceSupervisor().decide(refs,states,now,priority=priority)
                expected = ScalarReferenceSupervisor().decide(refs,states,now,priority=priority)
                assert actual == expected
                reasons.add(actual.reason)
    assert reasons == {'clear','priority-yield','brake-both','brake-both-unestablished'}
