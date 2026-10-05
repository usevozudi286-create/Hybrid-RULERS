# -*- coding: utf-8 -*-
"""
run_experiment.py — LIVE execution of the clean-repeatability experiment.

The default entry of this package is dry_run.py. This script makes PAID API
calls and refuses to run unless --live is passed explicitly. Nothing in this
package makes API calls on import, in tests, or in "model availability" checks.

Usage (author-confirmed budget items must be settled first, see
pending_confirmations.md):
  python scripts/run_experiment.py --live                 # repair off (main)
  python scripts/run_experiment.py --live --repair on     # repair sub-calls on

Frozen settings: temperature 0.3, max_tokens 16384, model = DEEPSEEK_MODEL env
or deepseek-v4-flash (see protocol.md S4 for the model-version uncertainty note).
Sequential execution (concurrency 1, matching the historical run).

Pre-run protocol revisions (recorded in protocol.md "Pre-run revisions"; the
frozen historical settings/logic are untouched):
  - repair: OFF by default for the main experiment (the historical chain had
    enable_repair=on; --repair on still exercises it explicitly).
  - explicit per-attempt HTTP timeout (default 120 s, --timeout to change);
    SDK-level retries disabled (max_retries=0): the outer retry policy owns
    all retries, with REAL backoff (time.sleep 2 s / 4 s).
  - the frozen JSON-mode fallback (an uncounted second request) is forbidden.
  - the send layer records the FULL request + response, slot_id, attempt_id
    and parameters for every attempt, and verifies each initial request
    against the frozen payload hash before sending.
  - resume verifies the run manifest (model / temperature / repair /
    simulated / thinking mode) and request hashes; mixing runs is refused.
    The original manifest is never overwritten.

Pre-run revision 8 (APPROVED by the author 2026-09-23 - approval_record.md):
  - --thinking-mode disabled sends the official extra_body
    {"thinking": {"type": "disabled"}}: deepseek-flash's thinking mode is ON
    by default and silently IGNORES temperature; only with it disabled does
    temperature=0.3 actually take effect. "default" sends no thinking
    parameter (the historical frozen behavior).
  - thinking_mode is recorded in every attempt's params and frozen in
    run_manifest.json; resuming with a different value is refused.

Budget guard (run-management ONLY, author-approved 2026-09-23, approval_record.md;
the scientific protocol is untouched):
  - before EVERY send and retry the guard checks the remaining budget and
    reserves the conservative full cost of the next attempt; an attempt
    whose usage is unknown is NEVER counted as 0 (imputed at the slot's full
    input estimate + max_tokens 16384 output, at PEAK prices).
  - if the projected cost would exceed the cap, it stops BEFORE sending:
    the checkpoint keeps all completed slots and the process exits with a
    budget report. No auto top-up, no silent cap raise.
  - state survives resume (attempts.jsonl is replayed before any send).
    Continuing after a budget stop requires an explicit author decision.

Outputs (results live under results_live/, keyed by opaque id only):
  results.jsonl / results_checkpoint.jsonl   one row per slot (resume-safe)
  attempts.jsonl                             every API attempt, full request
                                             and response, nothing discarded
  run_manifest.json                          written once; verified on resume
  run_info.json                              provider/model/temperature/max_tokens/
                                             timestamps - NO API keys, NO headers

Retry policy (protocol S5): per planned call at most 3 attempts (1 + 2 retries),
retries only on CONNECTION_ERROR / RATE_LIMIT_ERROR / 5xx server errors; the
first api_success response wins; parse failures / refusals / all-NA are never
re-asked. B's evidence-repair sub-call (when enabled) obeys the same policy.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from protocol_common import (  # noqa: E402
    PREP_DIR, TEMPERATURE, REQUEST_TIMEOUT_S, SDK_MAX_RETRIES, save_json,
)
import runner_core  # noqa: E402
from budget_guard import BudgetGuard, BudgetStop, BUDGET_CAP_YUAN  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="LIVE clean-repeatability run (paid API)")
    ap.add_argument("--live", action="store_true",
                    help="REQUIRED. Without this flag the script refuses to run.")
    ap.add_argument("--repair", choices=["on", "off"], default="off",
                    help="B evidence-repair sub-call. off (default, main "
                         "experiment - protocol revision): 1 call per B slot; "
                         "on: frozen historical logic (B slots may issue a "
                         "2nd call).")
    ap.add_argument("--model", default=None,
                    help="model override (default: DEEPSEEK_MODEL env or deepseek-v4-flash)")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S,
                    help="explicit per-attempt HTTP timeout in seconds "
                         "(default 120; recorded in the run manifest)")
    ap.add_argument("--thinking-mode", choices=["default", "disabled"],
                    default="default",
                    help="thinking mode (pre-run revision 8, author-approved "
                         "2026-09-23): 'default' sends no thinking parameter "
                         "(historical frozen behavior; the API default is "
                         "thinking ON, which silently ignores temperature). "
                         "'disabled' sends the official extra_body "
                         "{\"thinking\": {\"type\": \"disabled\"}} so that "
                         "temperature actually takes effect - the approved "
                         "live configuration. Recorded per attempt and "
                         "frozen in the run manifest.")
    ap.add_argument("--budget-cap", type=float, default=BUDGET_CAP_YUAN,
                    help="budget cap in CNY (run management only, "
                         "author-approved 2026-09-23: 100). Before every "
                         "send/retry the guard reserves the conservative "
                         "full cost of the next attempt; unknown usage is "
                         "never 0 (imputed at full slot input + 16384 "
                         "output, peak prices). Stops BEFORE sending when "
                         "the projection exceeds the cap; checkpoint keeps "
                         "all completed slots.")
    ap.add_argument("--out", default=os.path.join(PREP_DIR, "results_live"))
    args = ap.parse_args()

    if not args.live:
        print("REFUSING to run: this script makes PAID API calls.")
        print("The default entry of this package is scripts/dry_run.py (offline).")
        print("To start the real experiment you must explicitly pass --live.")
        sys.exit(2)

    enable_repair = (args.repair == "on")
    print("=" * 70)
    print("LIVE RUN - clean-repeatability experiment (1270 planned slots)")
    print("repair: %s | temperature: %s | timeout: %ss | sdk_max_retries: %d | "
          "thinking_mode: %s | budget_cap: ¥%.2f (run management only)"
          % ("on (frozen)" if enable_repair else "off (main, protocol revision)",
             args.temperature, args.timeout, SDK_MAX_RETRIES, args.thinking_mode,
             args.budget_cap))
    print("output: %s" % args.out)
    print("=" * 70)

    provider, info = runner_core.make_live_provider(args.temperature,
                                                    model_override=args.model,
                                                    timeout_s=args.timeout,
                                                    thinking_mode=args.thinking_mode)
    os.makedirs(args.out, exist_ok=True)
    save_json(os.path.join(args.out, "run_info.json"), {
        **info,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "enable_repair": enable_repair,
        "retry_policy": {
            "max_attempts_per_call": runner_core.MAX_ATTEMPTS_PER_CALL,
            "retryable": list(runner_core.RETRYABLE_ERROR_TYPES) +
                         ["API_ERROR with 'Server error' (5xx)"],
            "first_success_wins": True,
            "no_retry_on": ["parse failure", "refusal", "all-NA",
                            "auth/config errors", "json-mode errors (fallback forbidden)"],
            "backoff_s": runner_core.RETRY_BACKOFF_S,
            "backoff_real": True,   # time.sleep, not a no-op
        },
        "budget_guard": {
            "cap_yuan": args.budget_cap,
            "scope": "run management only (author-approved 2026-09-23); "
                     "no scientific-protocol change",
            "prices_used": "PEAK official prices (author-verified 2026-09-23): "
                           "input cache-miss ¥2/1M, output ¥8/1M",
            "unknown_usage": "never 0 - imputed at full slot input + 16384 "
                             "output, peak prices",
            "limits": "stop-on-local-projection, not a billing guarantee",
        },
        "note": "no API keys or Authorization headers are stored anywhere",
    })

    attempt_params = {
        "model": info["model"],
        "temperature": args.temperature,
        "max_tokens": 16384,
        "timeout_s": args.timeout,
        "sdk_max_retries": SDK_MAX_RETRIES,
        "response_format": {"type": "json_object"},
        "thinking_mode": args.thinking_mode,
    }

    guard = BudgetGuard(cap_yuan=args.budget_cap)
    budget_stopped = False
    try:
        queue, all_rows, executed, skipped, attempts = runner_core.run_queue(
            provider, enable_repair=enable_repair, resume=True, out_dir=args.out,
            simulated=False, quiet=False,
            sleep_fn=time.sleep,          # REAL backoff on retries (live only)
            attempt_params=attempt_params,
            budget_guard=guard)
    except BudgetStop as e:
        budget_stopped = True
        print("\n" + "=" * 70)
        print("BUDGET STOP (run management): %s" % e)
        print(json.dumps(guard.report(), ensure_ascii=False, indent=2))
        print("The checkpoint keeps every completed slot; nothing is lost.")
        print("Continuing requires an explicit author decision (review the")
        print("provider bill and authorize a new cap - the guard never")
        print("auto-tops-up). Resume with the SAME command after that.")
        print("=" * 70)
        save_json(os.path.join(args.out, "run_summary.json"), {
            "status": "budget_stopped",
            "budget": guard.report(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        sys.exit(3)
    save_json(os.path.join(args.out, "run_summary.json"), {
        "status": "completed",
        "executed_now": executed,
        "skipped_resumed": skipped,
        "unique_slot_rows": len(all_rows),
        "total_api_attempts": attempts,
        "budget": guard.report(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print("\nLIVE RUN finished: executed %d, resumed %d, unique rows %d, attempts %d"
          % (executed, skipped, len(all_rows), attempts))
    print("Budget accounting (conservative, peak prices): %s"
          % json.dumps(guard.report(), ensure_ascii=False))
    print("Analysis: python scripts/analyze.py --results %s/results.jsonl --out %s/analysis"
          % (args.out, args.out))


if __name__ == "__main__":
    main()
