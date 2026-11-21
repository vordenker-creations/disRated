from __future__ import annotations

import sys

__version__ = "0.1.2"

HELP_TEXT = """disRated: Evaluation harness and selective-classification benchmark for Trust & Safety moderation.

Usage:
  disrated <command> [options]

Core Commands:
  evalrun, run              Run model inference over dataset splits
  score                     Score predictions against ground truth labels and verify provenance
  val-gate                  Run Phase 3 validation gate, calibration, and operating-point lock
  eval-test                 Run Phase 4 held-out test evaluation from a locked validation gate
  probe                     Train and evaluate Tier 0 linear probe on cached embeddings
  embed                     Extract and cache vision embeddings for dataset images

Analysis & Audits:
  risk-coverage             Evaluate one-knob risk-coverage tradeoff with bootstrap CIs
  sweep-thresholds          Search two-knob operating thresholds
  rebuild-splits            Rebuild graph-isolated connected-component splits
  audit-hashes              Audit pairwise 64-bit pHash & dHash Hamming distance across splits
  audit-clip                Audit pairwise CLIP cosine similarity across splits
  audit-components          Inspect connected-component size distribution
  audit-duplicate-labels    Check label consistency among duplicate image clusters
  audit-dhash-residuals     Inspect dHash residual distribution
  audit-effect              Compute leakage effect size via paired bootstrap
"""


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(HELP_TEXT)
        return

    if argv[0] in ("-V", "--version"):
        print(f"disrated {__version__}")
        return

    cmd = argv[0].replace("_", "-")
    cmd_args = argv[1:]
    sys.argv = [f"disrated {argv[0]}", *cmd_args]

    if cmd in ("evalrun", "run"):
        from .evalrun import main as sub_main

        sub_main(cmd_args)
    elif cmd == "score":
        from .score import main as sub_main

        sub_main(cmd_args)
    elif cmd in ("val-gate", "val_gate"):
        from .val_gate import main as sub_main

        sub_main(cmd_args)
    elif cmd in ("eval-test", "eval_test"):
        from .eval_test import main as sub_main

        sub_main(cmd_args)
    elif cmd == "probe":
        from .probe import main as sub_main

        sub_main(cmd_args)
    elif cmd == "embed":
        from .embed import main as sub_main

        sub_main(cmd_args)
    elif cmd in ("risk-coverage", "risk_coverage"):
        from .risk_coverage import main as sub_main

        sub_main(cmd_args)
    elif cmd in ("sweep-thresholds", "sweep_thresholds"):
        from .sweep_thresholds import main as sub_main

        sub_main()
    elif cmd in ("rebuild-splits", "rebuild_splits"):
        from .rebuild_splits import main as sub_main

        sub_main()
    elif cmd in ("audit-hashes", "audit_hashes"):
        from .audit_hashes import main as sub_main

        sub_main()
    elif cmd in ("audit-clip", "audit_clip"):
        from .audit_clip import main as sub_main

        sub_main()
    elif cmd in ("audit-components", "audit_components"):
        from .audit_components import main as sub_main

        sub_main()
    elif cmd in ("audit-duplicate-labels", "audit_duplicate_labels"):
        from .audit_duplicate_labels import main as sub_main

        sub_main()
    elif cmd in ("audit-dhash-residuals", "audit_dhash_residuals"):
        from .audit_dhash_residuals import main as sub_main

        sub_main()
    elif cmd in ("audit-effect", "audit_effect"):
        from .audit_effect import main as sub_main

        sub_main()
    else:
        print(f"Unknown command: {argv[0]}\n")
        print(HELP_TEXT)
        sys.exit(1)
