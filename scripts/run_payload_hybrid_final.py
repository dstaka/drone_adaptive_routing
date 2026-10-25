#!/usr/bin/env python3
"""C-only final development, validation, freeze and once-only formal execution.

Never selects a controller from final outcomes. Every physical case has its own
exclusive directory; an interrupted attempt must be audited, never overwritten.
"""

import argparse
import json
from pathlib import Path

from adaptive_drone_patrol.jobs import write_json
from adaptive_drone_patrol.payload_hybrid_final import (
    FinalContext,
    FinalFreeze,
    prepare_development,
    _development_declaration,
    run_final_case,
    prepare_formal,
    digest,
    write_reader_rebinding,
)
from adaptive_drone_patrol.payload_hybrid_flight import ReferenceAdmission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare-development",
            "reader-rebind",
            "robustness",
            "validation",
            "freeze",
            "prepare-formal",
            "formal",
            "audit",
        ),
    )
    parser.add_argument("--reference-rebinding", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Development campaign directory, formal campaign directory, or new freeze JSON for freeze mode",
    )
    parser.add_argument("--development", type=Path)
    parser.add_argument("--reader-rebinding", type=Path)
    parser.add_argument("--old-source-package", type=Path)
    parser.add_argument("--diagnostic-roots", type=Path, nargs="+")
    parser.add_argument("--duration-root", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument(
        "--case-id", help="Execute exactly this declared case; existing attempts are rejected"
    )
    args = parser.parse_args(argv)
    reference = ReferenceAdmission.from_rebinding(json.loads(args.reference_rebinding.read_text()))
    context = FinalContext.from_reference(
        reference,
        design=args.design,
        original=args.original,
        routing=args.routing,
        reader_rebinding=args.reader_rebinding,
    )
    if args.mode == "reader-rebind":
        if args.development is None or args.old_source_package is None or args.reader_rebinding is not None:
            parser.error(
                "reader-rebind requires --development and --old-source-package, without --reader-rebinding"
            )
        record = write_reader_rebinding(
            args.output,
            context=context,
            old_development=args.development,
            old_source_package=args.old_source_package,
        )
        print("READER_DERIVATION", digest(record), "NO_PHYSICS_OR_FORMAL_COMPATIBILITY", flush=True)
        return
    if args.mode == "prepare-development":
        prepare_development(args.output, context)
        return
    from adaptive_drone_patrol.routing_agent import RoutingAgent
    from adaptive_drone_patrol.payload_hybrid_final import ROUTING_SHA256

    planner = RoutingAgent(args.routing, expected_hash=ROUTING_SHA256)
    if args.mode in ("robustness", "validation"):
        declaration = _development_declaration(args.output, context)
        key = "robustness_cases" if args.mode == "robustness" else "validation_cases"
        cases = declaration[key]
        stage = args.mode
        (args.output / stage).mkdir(exist_ok=True)
        freeze = None
    elif args.mode == "freeze":
        if args.development is None or not args.diagnostic_roots or args.duration_root is None:
            parser.error("freeze requires --development, --diagnostic-roots and --duration-root")
        if args.output.exists():
            raise FileExistsError(args.output)
        frozen = FinalFreeze.create(
            context,
            development=args.development,
            diagnostic_roots=args.diagnostic_roots,
            duration_root=args.duration_root,
            planner=planner,
        )
        write_json(args.output, frozen.record)
        print("FINAL_FREEZE", digest(frozen.record), "FORMAL_INPUTS_UNOPENED", flush=True)
        return
    else:
        if args.freeze is None:
            parser.error("formal/audit modes require --freeze")
        freeze = FinalFreeze.load(args.freeze, context, planner)
        if args.mode == "prepare-formal":
            prepare_formal(args.output, freeze)
            return
        if args.mode == "audit":
            from adaptive_drone_patrol.payload_hybrid_final_audit import audit_final_campaign

            report = audit_final_campaign(freeze, args.output, planner)
            write_json(args.output / "final-independent-audit.json", report)
            print("FINAL_ACCEPTANCE", report["accepted"], flush=True)
            return
        from adaptive_drone_patrol.payload_hybrid_final import require_formal_declaration

        cases = require_formal_declaration(args.output, freeze)["cases"]
        stage = None
    if args.case_id:
        cases = [c for c in cases if c["case_id"] == args.case_id]
        if len(cases) != 1:
            parser.error("--case-id must identify one case in this declaration")
    for case in cases:
        target = args.output / stage / case["case_id"] if stage else args.output / case["case_id"]
        checked = run_final_case(context, case, target, planner, freeze=freeze)
        print(
            "FINAL_CASE",
            case["case_id"],
            checked["success"],
            checked["completed_targets"],
            checked["terminal_flush_wall_s"],
            checked["audit_errors"],
            flush=True,
        )
    if stage is None and not args.case_id:
        from adaptive_drone_patrol.payload_hybrid_final_audit import audit_final_campaign

        report = audit_final_campaign(freeze, args.output, planner)
        write_json(args.output / "final-independent-audit.json", report)
        print("FINAL_ACCEPTANCE", report["accepted"], flush=True)


if __name__ == "__main__":
    main()
