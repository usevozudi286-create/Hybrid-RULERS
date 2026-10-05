# -*- coding: utf-8 -*-
"""
runner_core.py — shared execution core for the clean-request repeatability
experiment. Used by dry_run.py (default entry, simulated, no network) and
run_experiment.py (--live only, paid API calls).

Frozen behavior implemented here:
  - One planned call slot per (method, sample, repeat); 1270 slots total.
  - Retry policy (protocol S5): at most MAX_ATTEMPTS_PER_CALL=3 attempts per
    planned API call (1 + up to 2 retries). Retries ONLY on
    CONNECTION_ERROR / RATE_LIMIT_ERROR / server 5xx (API_ERROR with
    'Server error'). The FIRST api_success=True response wins. Parse failures,
    refusals, empty responses and all-NA outputs are NOT retried.
    Every attempt is logged (attempts.jsonl), nothing is discarded.
  - B (Prompt RULERS): frozen score_single_sample() with the locked EN rubric,
    enable_repair per configuration (historical frozen logic had it ON).
    The repair sub-call goes through the same retry policy.
  - A (Direct, cleaned): strict raw-output validation on the CLEANED messages
    from the offline payloads (opaque id, full text). Protocol revision: the
    historical default-0/clamp rule is NOT applied to new results; the main
    label comes from a VALID risk_score at THRESHOLD_DIRECT (see
    _validate_direct_output).
  - B responses are structurally validated BEFORE the frozen chain
    (StructuralGateProvider); structural invalidity => risk MISSING, while
    evidence-VERIFICATION failures keep their scores and are flagged
    separately.
  - Checkpoint/resume keyed by slot_id: re-running never double-counts.
  - Results are keyed by opaque_id ONLY; the real id mapping never enters
    results or requests.

NO network calls on import. NO API keys are stored.
"""

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _SCRIPT_DIR)

from protocol_common import (  # noqa: E402
    PREP_DIR, MAX_ATTEMPTS_PER_CALL, RETRYABLE_ERROR_TYPES,
    REPAIR_DEFAULT, REQUEST_TIMEOUT_S, SDK_MAX_RETRIES,
    THINKING_EXTRA_BODY, THRESHOLD_DIRECT, THRESHOLD_PROMPT_RULERS, ALL_CIDS,
    get_locked_rubric_bundle, sha256_obj, save_json,
)

REQUESTS_DIR = os.path.join(PREP_DIR, "requests")

RETRY_BACKOFF_S = [2.0, 4.0]  # backoff before retry 1 / retry 2 (live mode only)


# ---------------------------------------------------------------------------
# Queue / payload loading
# ---------------------------------------------------------------------------

def load_queue():
    rows = []
    with open(os.path.join(REQUESTS_DIR, "queue.csv"), "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    assert len(rows) == 1270, "queue has %d slots, expected 1270" % len(rows)
    return rows


def load_payload(method, opaque_id):
    letter = "A" if method == "direct" else "B"
    path = os.path.join(REQUESTS_DIR, "payloads", "PAY_%s_%s.json" % (letter, opaque_id))
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    return p["messages"], p["payload_sha256"]


def load_selected_texts():
    with open(os.path.join(PREP_DIR, "text_freeze", "selected_texts.json"),
              "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Retry policy wrapper
# ---------------------------------------------------------------------------

def _is_retryable(error_type, error_message):
    if error_type in RETRYABLE_ERROR_TYPES:
        return True
    if error_type == "API_ERROR" and "Server error" in (error_message or ""):
        return True
    return False


def _is_repair_messages(messages):
    """Repair prompts carry the '"repair_status"' output key; the scoring
    prompt never does. (Substring "repairs" is NOT safe: the EN rubric
    contains "self-repairs".)"""
    return any('"repair_status"' in (m.get("content") or "")
               for m in messages if m.get("role") == "user")


class RetryingProvider:
    """
    Wraps any provider exposing generate_json(messages) -> standard response
    dict, applying the frozen retry policy per planned call. Records every
    attempt. Never raises for classification-level errors (mirrors frozen
    provider behavior: errors become response dicts).

    Protocol-revision additions (additive, send layer only):
      - Before EACH attempt is sent, the request is hashed and compared with
        the frozen payload_sha256 of the current slot. A mismatch REFUSES to
        send (RuntimeError). Repair sub-call messages are dynamically built
        by the frozen chain and are exempt from this check (they are still
        recorded with their own hash).
      - Every attempt record carries slot_id / attempt_id / the FULL request
        messages + parameters + the FULL response dict; nothing is curated
        away at the send layer (the frozen chain's own curation is untouched,
        this log is separate).
      - pre_send_check (optional, run management only): called BEFORE every
        attempt (planned attempt and each retry), after the frozen-payload
        verification and outside the transport try/except, so a BudgetStop
        raised there propagates and halts the run cleanly. None by default
        (dry run unchanged).
    """

    def __init__(self, inner, sleep_fn=None, record_attempt=None,
                 pre_send_check=None):
        self.inner = inner
        self._sleep = sleep_fn or (lambda s: None)
        self._record = record_attempt or (lambda rec: None)
        self._pre_send_check = pre_send_check
        self.total_attempts = 0
        # Per-slot context, set by run_queue before each slot.
        self.current_slot_id = None
        self.current_expected_sha256 = None
        self.current_params = None
        # Structural-gate diagnostics (B slots only): exposed from the inner
        # provider when it is a StructuralGateProvider, else None.
        self.diag_history = getattr(inner, "diag_history", None)
        # In-order log of EVERY attempt across ALL planned calls of a run.
        # The frozen score_single_sample() curates provider_raw and drops the
        # attempt list, so callers slice this history around the frozen chain
        # to recover per-slot attempts (first-pass + repair sub-call). The
        # history keeps the meta record; the FULL request/response record goes
        # to the attempts log (record_attempt) to bound memory.
        self.attempt_history = []

    def _verify_frozen_payload(self, messages):
        if self.current_expected_sha256 is None:
            return
        if _is_repair_messages(messages):
            return
        actual = sha256_obj(messages)
        if actual != self.current_expected_sha256:
            raise RuntimeError(
                "PAYLOAD MISMATCH - refusing to send: slot %s request hash %s "
                "!= frozen payload %s"
                % (self.current_slot_id, actual[:16],
                   self.current_expected_sha256[:16]))

    def generate_json(self, messages):
        attempts = []
        resp = None
        call_kind = "repair" if _is_repair_messages(messages) else "scoring"
        for idx in range(MAX_ATTEMPTS_PER_CALL):
            self._verify_frozen_payload(messages)  # refuse BEFORE sending
            if self._pre_send_check is not None:
                # Budget guard (run management only): BEFORE every send and
                # retry. Deliberately OUTSIDE the try below so BudgetStop
                # propagates and halts the run (nothing is sent).
                self._pre_send_check(self.current_slot_id, idx, call_kind)
            t0 = time.perf_counter()
            try:
                resp = self.inner.generate_json(messages)
            except Exception as e:  # unexpected transport-level raise
                resp = {
                    "provider": getattr(self.inner, "provider_name", "?"),
                    "model": getattr(self.inner, "model", "?"),
                    "raw_text": "",
                    "parsed_json": None,
                    "parse_success": False,
                    "api_success": False,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "error_type": "API_ERROR",
                    "error_message": str(e)[:500],
                    "simulated_variation": False,
                }
            self.total_attempts += 1
            # FULL send-layer record: slot_id, attempt_id, complete request
            # (messages + params + hash) and complete response. Nothing is
            # curated away here.
            full_rec = {
                "slot_id": self.current_slot_id,
                "attempt_id": ("%s-a%d" % (self.current_slot_id, idx)
                               if self.current_slot_id else "unknown-a%d" % idx),
                "attempt_idx": idx,
                "call_kind": call_kind,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request": {
                    "messages": messages,
                    "params": self.current_params or {},
                    "request_sha256": sha256_obj(messages),
                },
                "response": {k: v for k, v in resp.items() if k != "attempts"},
            }
            rec = {k: v for k, v in full_rec.items() if k not in ("request", "response")}
            rec.update({
                "api_success": resp.get("api_success", False),
                "parse_success": resp.get("parse_success", False),
                "error_type": resp.get("error_type"),
                "error_message": resp.get("error_message"),
                "latency_ms": resp.get("latency_ms"),
                "response_model": resp.get("model"),
                "finish_reason": resp.get("finish_reason"),
                "usage": resp.get("usage"),
            })
            attempts.append(rec)
            self._record(full_rec)
            self.attempt_history.append(rec)
            if resp.get("api_success"):
                break  # first successful API response wins
            if not _is_retryable(resp.get("error_type"), resp.get("error_message")):
                break  # parse failures / auth / config / other -> no retry
            if idx + 1 < MAX_ATTEMPTS_PER_CALL:
                self._sleep(RETRY_BACKOFF_S[min(idx, len(RETRY_BACKOFF_S) - 1)])
        resp["attempts"] = attempts
        return resp


# ---------------------------------------------------------------------------
# Raw-output validation (protocol revisions, additive; the frozen historical
# rules are recorded in protocol.md and are NOT applied to new results)
# ---------------------------------------------------------------------------

def _strict_bounded(value, lo, hi):
    """Strict numeric validation for a bounded value. Returns (value|None,
    reason). Bools (True==1) are invalid, non-finite and out-of-range values
    are invalid: the result is MISSING, never defaulted, never clamped."""
    if value is None:
        return None, "missing"
    if isinstance(value, bool):
        return None, "bool"
    if not isinstance(value, (int, float)):
        return None, "non_numeric"
    f = float(value)
    if not math.isfinite(f):
        return None, "non_finite"
    if f < lo or f > hi:
        return None, "out_of_range"
    return f, "ok"


def _json_safe_raw(v):
    """Make a raw reported value JSON-storable for diagnostics (NaN/Inf stay
    visible as strings instead of corrupting the JSONL)."""
    if v is None or isinstance(v, (str, int)):
        return v
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            return "NaN" if math.isnan(v) else ("Infinity" if v > 0 else "-Infinity")
        return v
    if isinstance(v, dict):
        return {str(k): _json_safe_raw(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe_raw(x) for x in v]
    return str(v)


def _json_structure_diagnostics(raw_text):
    """
    Diagnostic scan of a raw JSON text (never raises; parsing itself stays
    with the frozen provider):
      - duplicate keys at ANY nesting level (json.loads keeps the LAST value;
        the keys are recorded so it is never silent)
      - the parsed top-level object (for unknown-field reporting)
    """
    duplicates = set()

    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                duplicates.add(str(k))
            out[k] = v
        return out

    top = None
    try:
        data = json.loads(raw_text, object_pairs_hook=hook)
        if isinstance(data, dict):
            top = data
    except Exception:
        return sorted(duplicates), None
    return sorted(duplicates), top


DIRECT_KNOWN_FIELDS = {"sample_id", "predicted_label", "risk_score",
                       "confidence", "evidence", "reason"}


def _validate_direct_output(parsed, parse_success, raw_text=""):
    """
    Protocol-revision Direct (A) validation for the MAIN experiment:
      - risk_score is strictly validated (bool / non-numeric / non-finite /
        out-of-range -> risk MISSING). It is NEVER defaulted to 0 and NEVER
        clamped into range.
      - predicted_label is kept RAW for diagnostics only; the main-analysis
        label is computed from the VALID risk at THRESHOLD_DIRECT.
      - duplicate JSON keys and unknown top-level fields are recorded, never
        silently accepted.
    """
    out = {
        "risk_valid": False,
        "risk_reason": "missing",
        "risk_score": None,             # main analysis (validated only)
        "risk_score_raw": None,         # diagnostic: exact reported value
        "predicted_label_raw": None,    # diagnostic only
        "label_direct": None,           # main-analysis label from valid risk
        "confidence": None,
        "confidence_reason": "missing",
        "unknown_fields": [],
        "duplicate_fields": [],
    }
    if not parse_success or not isinstance(parsed, dict):
        return out
    risk, reason = _strict_bounded(parsed.get("risk_score"), 0.0, 1.0)
    out["risk_score_raw"] = _json_safe_raw(parsed.get("risk_score"))
    out["risk_valid"] = risk is not None
    out["risk_reason"] = reason
    if risk is not None:
        out["risk_score"] = round(risk, 4)
        out["label_direct"] = "positive" if risk >= THRESHOLD_DIRECT else "control"
    out["predicted_label_raw"] = _json_safe_raw(parsed.get("predicted_label"))
    conf, creason = _strict_bounded(parsed.get("confidence"), 0.0, 1.0)
    out["confidence"] = conf
    out["confidence_reason"] = creason
    dupes, top = _json_structure_diagnostics(raw_text)
    out["duplicate_fields"] = dupes
    if isinstance(top, dict):
        out["unknown_fields"] = sorted(set(top.keys()) - DIRECT_KNOWN_FIELDS)
    return out


# ---------------------------------------------------------------------------
# B structural gate (protocol revision, additive; the frozen chain is
# untouched and is shielded from malformed structures it would crash on)
# ---------------------------------------------------------------------------

RULERS_KNOWN_FIELDS = {"sample_id", "checklist", "trait_scores", "evidence",
                       "confidence", "uncertainty", "human_review_recommended",
                       "reason"}
REPAIR_KNOWN_FIELDS = {"repairs"}


def _checklist_item_ok(item):
    if not isinstance(item, dict):
        return False, "checklist item is not an object"
    cid = item.get("criterion_id")
    if not isinstance(cid, str) or not cid:
        return False, "checklist item criterion_id missing or not a string"
    score = item.get("score")
    if isinstance(score, bool) or score not in (0, 1, 2, "NA"):
        return False, "checklist item score not in {0,1,2,'NA'}"
    return True, ""


def validate_rulers_structure(parsed):
    """
    Structural validation of a B scoring response (BEFORE the frozen chain
    sees it). Structural invalidity => the response is treated as a parse
    failure and the slot's risk is MISSING. Evidence-VERIFICATION failures
    are NOT structural: their structure is fine, scores are kept, and they
    are reported separately (see run_slot_rulers).
    """
    errors = []
    if not isinstance(parsed, dict):
        return False, ["parsed JSON is not an object"]
    checklist = parsed.get("checklist")
    if not isinstance(checklist, list) or not checklist:
        return False, ["checklist missing or not a non-empty list"]
    seen = set()
    for item in checklist:
        ok, msg = _checklist_item_ok(item)
        if not ok:
            errors.append(msg)
            continue
        cid = item["criterion_id"]
        if cid in seen:
            errors.append("duplicate criterion_id %s in checklist" % cid)
        seen.add(cid)
    for cid in ALL_CIDS:
        if cid not in seen:
            errors.append("missing criterion_id %s in checklist" % cid)
    trait_scores = parsed.get("trait_scores")
    if trait_scores is not None:
        if not isinstance(trait_scores, dict):
            errors.append("trait_scores is not an object")
        else:
            for k, v in trait_scores.items():
                if v == "NA" or v is None:
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errors.append("trait_scores.%s not numeric/NA" % k)
    evidence = parsed.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, list):
            errors.append("evidence is not a list")
        else:
            for ev in evidence:
                # the frozen chain does ev.get("criterion_id") and would crash
                # on a string item; dicts are the structural minimum
                if not isinstance(ev, dict):
                    errors.append("evidence item is not an object")
    return (len(errors) == 0), errors


def validate_repair_structure(parsed):
    errors = []
    if not isinstance(parsed, dict):
        return False, ["repair JSON is not an object"]
    repairs = parsed.get("repairs")
    if not isinstance(repairs, list):
        return False, ["repairs missing or not a list"]
    for rp in repairs:
        if not isinstance(rp, dict):
            errors.append("repair item is not an object")
            continue
        if not isinstance(rp.get("criterion_id"), str):
            errors.append("repair item criterion_id missing or not a string")
        if not isinstance(rp.get("repair_status"), str):
            errors.append("repair item repair_status missing or not a string")
    return (len(errors) == 0), errors


class StructuralGateProvider:
    """
    Protocol-revision wrapper for B slots. Validates the STRUCTURE of each
    parsed response before the frozen chain sees it:
      - structurally invalid -> rewritten into a parse failure
        (parse_success=False, error_type=STRUCTURAL_INVALID, api_success kept)
        so the frozen chain is shielded and the slot ends with risk MISSING;
      - duplicate keys / unknown top-level fields recorded per call;
      - evidence-verification failures pass through untouched (they are
        reported separately by run_slot_rulers, scores kept).
    Repair sub-calls are detected via the '"repair_status"' marker and
    validated against the repair schema.
    """
    def __init__(self, inner):
        self.inner = inner
        self.diag_history = []  # per call, sliced around the frozen chain

    def generate_json(self, messages):
        is_repair = _is_repair_messages(messages)
        resp = self.inner.generate_json(messages)
        diag = {
            "call_kind": "repair" if is_repair else "scoring",
            "duplicate_fields": [],
            "unknown_fields": [],
            "structural_errors": [],
            "structural_invalid": False,
        }
        if resp.get("parse_success") and isinstance(resp.get("parsed_json"), dict):
            dupes, top = _json_structure_diagnostics(resp.get("raw_text", ""))
            diag["duplicate_fields"] = dupes
            known = REPAIR_KNOWN_FIELDS if is_repair else RULERS_KNOWN_FIELDS
            if isinstance(top, dict):
                diag["unknown_fields"] = sorted(set(top.keys()) - known)
            if is_repair:
                ok, errors = validate_repair_structure(resp["parsed_json"])
            else:
                ok, errors = validate_rulers_structure(resp["parsed_json"])
            if not ok:
                diag["structural_invalid"] = True
                diag["structural_errors"] = errors
                resp = dict(resp)  # never mutate the inner provider's dict
                resp.update({
                    "parsed_json": None,
                    "parse_success": False,
                    # api_success stays True: the API call itself succeeded;
                    # the OUTPUT structure failed, so the slot is unscorable.
                    "error_type": "STRUCTURAL_INVALID",
                    "error_message": "structure invalid: " + "; ".join(errors)[:400],
                })
        self.diag_history.append(diag)
        return resp


def _flags_from_rulers_result(score_result, rubric_checklist_n=13):
    """
    New-protocol flags (protocol S5), computed ON TOP of the frozen chain.
    Never changes the frozen stored values; the analyzer consumes the flags.
    """
    parsed = score_result.get("parsed_output")
    scores = score_result.get("scores", {})
    checklist = scores.get("checklist_scores", {})
    n_items = rubric_checklist_n
    non_na = {c: v for c, v in checklist.items() if v in (0, 1, 2)}
    na = {c: v for c, v in checklist.items() if v == "NA" or v is None}
    all_na_13 = len(na) == n_items and len(non_na) == 0
    six_cids = {"C01", "C02", "C07", "C08", "C09", "C10"}
    six_valid = {c for c in six_cids if checklist.get(c) in (0, 1, 2)}
    six_na = len(six_valid) == 0
    return {
        "all_na_13": all_na_13,
        "six_llm_criteria_all_na": six_na,
        "n_valid_scores": len(non_na),
        "n_na_scores": len(na),
    }


def run_slot_direct(messages, provider, text_sha256):
    """
    A slot: cleaned Direct request + strict risk validation (protocol
    revision). The main-analysis label is risk>=THRESHOLD_DIRECT on a VALID
    risk score; the raw predicted_label is kept for diagnostics only.
    """
    resp = provider.generate_json(messages)
    parsed = resp.get("parsed_json") or {}
    parse_success = resp.get("parse_success", False)
    v = _validate_direct_output(parsed, parse_success, resp.get("raw_text", ""))
    return {
        "api_success": resp.get("api_success", False),
        "parse_success": parse_success,
        "error_type": resp.get("error_type"),
        "error_message": resp.get("error_message"),
        **v,
        "output_json": parsed if parse_success else None,
        "attempts": resp.get("attempts", []),
        "latency_ms": resp.get("latency_ms"),
        "response_model": resp.get("model"),
        "finish_reason": resp.get("finish_reason"),
        "usage": resp.get("usage"),
        "text_sha256": text_sha256,
    }


def run_slot_rulers(text, provider, enable_repair, calibrator=None):
    """
    B slot: frozen score_single_sample() chain on the canonical text
    (no sample id in the prompt - frozen template behavior), plus protocol
    flags and the structural-gate diagnostics.

    Dispositions (protocol revision, distinct and never conflated):
      - STRUCTURAL invalidity (gate): risk MISSING, slot unscorable,
        structural_invalid=True. Never 0, never clamped.
      - Evidence-VERIFICATION failure (frozen verifier, structure fine):
        scores KEPT, slot scorable, evidence_verification_failed=True,
        reported separately.
      - all-13-NA: risk MISSING (protocol S5), never 0.
    """
    from mci_rulers_core import score_single_sample
    from mci_calibration import NoOpCalibrator

    if calibrator is None:
        calibrator = NoOpCalibrator()
    bundle = get_locked_rubric_bundle()
    sample = {"sample_id": "unknown", "text": text}
    # The frozen score_single_sample() curates provider_raw and drops the
    # attempt log, so snapshot the RetryingProvider history around the frozen
    # chain to recover this slot's attempts (first-pass + repair sub-call,
    # each with usage / finish_reason / timestamps). Same for the structural
    # gate's per-call diagnostics.
    before = len(provider.attempt_history) \
        if hasattr(provider, "attempt_history") else 0
    diag_before = len(provider.diag_history) if provider.diag_history else 0
    result = score_single_sample(
        sample, bundle, provider, calibrator,
        text_field="text", id_field="sample_id", enable_repair=enable_repair,
    )
    slot_attempts = (provider.attempt_history[before:]
                     if hasattr(provider, "attempt_history") else [])
    slot_diags = (provider.diag_history[diag_before:]
                  if provider.diag_history else [])
    raw = result.get("provider_raw", {})
    scores = result.get("scores", {})
    parsed = result.get("parsed_output", {})
    parse_success = bool(result.get("parse_success"))
    last_attempt = slot_attempts[-1] if slot_attempts else {}
    evidence_repair = result.get("evidence_repair") or {}

    structural_errors = []
    repair_structural_errors = []
    duplicate_fields = []
    unknown_fields = []
    for d in slot_diags:
        duplicate_fields += d.get("duplicate_fields", [])
        unknown_fields += d.get("unknown_fields", [])
        if d.get("structural_invalid"):
            if d.get("call_kind") == "repair":
                # repair sub-call structurally invalid: the FIRST-PASS scores
                # are still valid; slot stays scorable, flagged separately
                repair_structural_errors += d.get("structural_errors", [])
            else:
                structural_errors += d.get("structural_errors", [])
    structural_invalid = bool(structural_errors)

    if parse_success and not structural_invalid:
        flags = _flags_from_rulers_result(result)
        raw_risk = scores.get("raw_risk_score", 0.0)
        if flags["all_na_13"]:
            # protocol S5: all-13-NA -> risk MISSING, never 0
            risk_score = None
            legacy_label = None
        else:
            # Frozen normalization chain (stability_experiment.py
            # run_rulers_sample); the scoring rule itself is untouched.
            normalized_risk = max(0.0, min(1.0, raw_risk / 2.0))
            risk_score = round(normalized_risk, 4)
            legacy_label = "positive" if normalized_risk >= 0.5 else "control"
    else:
        flags = {"all_na_13": False, "six_llm_criteria_all_na": False,
                 "n_valid_scores": 0, "n_na_scores": 0}
        risk_score = None
        legacy_label = None

    label_rulers = None
    if risk_score is not None:
        label_rulers = "positive" if risk_score >= THRESHOLD_PROMPT_RULERS else "control"

    # Evidence-verification failure: structure was fine and scores exist, but
    # the frozen verifier rejected evidence. Distinct from structural
    # invalidity: kept + flagged, never dropped.
    evidence_verification_failed = bool(
        parse_success and not structural_invalid
        and evidence_repair.get("final_valid") is False)

    confidence = parsed.get("confidence", None) if isinstance(parsed, dict) else None
    return {
        "api_success": raw.get("api_success", False),
        "parse_success": parse_success,
        "error_type": raw.get("error_type"),
        "error_message": raw.get("error_message"),
        "raw_risk_score": scores.get("raw_risk_score"),
        "risk_score": risk_score,
        "label_rulers": label_rulers,
        "legacy_label_0p5": legacy_label,   # historical artifact, NOT a protocol threshold
        "structural_invalid": structural_invalid,
        "structural_errors": structural_errors,
        "repair_structural_invalid": bool(repair_structural_errors),
        "repair_structural_errors": repair_structural_errors,
        "evidence_verification_failed": evidence_verification_failed,
        "duplicate_fields": sorted(set(duplicate_fields)),
        "unknown_fields": sorted(set(unknown_fields)),
        "confidence": confidence,
        "output_json": parsed if parse_success else None,
        "checklist_scores": scores.get("checklist_scores", {}),
        "trait_scores": scores.get("trait_scores", {}),
        "evidence_repair": evidence_repair,
        "flags": flags,
        "attempts": slot_attempts or raw.get("attempts", []),
        "latency_ms": result.get("timing_ms"),
        "response_model": raw.get("model"),
        "finish_reason": raw.get("finish_reason") or last_attempt.get("finish_reason"),
        "usage": raw.get("usage") or last_attempt.get("usage"),
        "rubric_hash": result.get("rubric_hash"),
    }


# ---------------------------------------------------------------------------
# Slot result assembly
# ---------------------------------------------------------------------------

def assemble_slot_row(slot, result):
    return {
        "slot_id": slot["slot_id"],
        "method": slot["method"],
        "repeat_idx": int(slot["repeat_idx"]),
        "opaque_id": slot["opaque_id"],
        "payload_sha256": slot["payload_sha256"],
        "text_sha256": slot["text_sha256"],
        "simulated": result.get("simulated", False),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }


def load_checkpoint(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["slot_id"]] = r
    return done


def _file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _verify_frozen_requests(queue):
    """Re-hash every frozen payload file against the queue's expected hashes
    (254 unique payloads). Runs BEFORE any slot can be sent."""
    seen = set()
    for q in queue:
        key = (q["method"], q["opaque_id"])
        if key in seen:
            continue
        seen.add(key)
        msgs, _psha = load_payload(q["method"], q["opaque_id"])
        assert sha256_obj(msgs) == q["payload_sha256"], \
            "frozen payload mismatch: %s" % q["slot_id"]


def run_queue(provider, enable_repair=None, resume=True, out_dir=None,
              simulated=False, max_slots=None, quiet=False, slot_hook=None,
              sleep_fn=None, attempt_params=None, budget_guard=None):
    """
    Execute the frozen queue sequentially. Returns list of slot rows.
    - resume: skip slots already present in results_checkpoint.jsonl
    - every completed slot appended to checkpoint + results.jsonl + attempts.jsonl
    - duplicate slot execution is impossible by construction (checkpoint gate);
      the final assertion enforces exactly one row per slot.

    Protocol-revision additions (pre-run fixes, additive):
      - run_manifest.json: written once (never overwritten). On resume, the
        stored config (model / temperature / repair / simulated) is compared
        with the current run and any mixing is REFUSED; queue.csv and the
        payload manifest hashes are re-verified too.
      - All 254 frozen payload files are re-hashed against the queue BEFORE
        anything is sent.
      - enable_repair defaults to REPAIR_DEFAULT (main experiment: repair=off).
      - sleep_fn (live: time.sleep) gives the outer retry REAL backoff.
      - attempt_params (model/temperature/max_tokens/timeout/sdk retries/
        thinking_mode) are recorded on every attempt and written into the
        manifest; thinking_mode is a frozen config key (revision 8, pending).
      - budget_guard (optional, run management only, author-approved
        2026-09-23): when given, its state is replayed from attempts.jsonl
        (resume-safe), every completed attempt is accounted, and
        check_before_send is invoked BEFORE every send/retry; BudgetStop
        propagates out (the checkpoint keeps all completed slots). None by
        default (dry run unchanged).
    """
    out_dir = out_dir or PREP_DIR
    enable_repair = REPAIR_DEFAULT if enable_repair is None else enable_repair
    results_path = os.path.join(out_dir, "results.jsonl")
    checkpoint_path = os.path.join(out_dir, "results_checkpoint.jsonl")
    attempts_path = os.path.join(out_dir, "attempts.jsonl")
    manifest_path = os.path.join(out_dir, "run_manifest.json")

    # Idempotency guard (protocol S5): a fresh run (resume=False) into a
    # directory that already holds checkpoint rows is refused BEFORE any slot
    # is executed, so a re-run can never append duplicate slots. Continue with
    # resume=True, or move/delete the directory to start over.
    if not resume and os.path.exists(checkpoint_path) \
            and os.path.getsize(checkpoint_path) > 0:
        raise AssertionError(
            "fresh run refused: %s already contains checkpoint rows; pass "
            "resume=True to continue, or move/delete the directory to start "
            "over" % checkpoint_path)

    queue = load_queue()
    texts = load_selected_texts()
    done = load_checkpoint(checkpoint_path) if resume else {}

    # ---- run manifest: write once, verify on resume, never overwrite ------
    queue_sha = _file_sha256(os.path.join(REQUESTS_DIR, "queue.csv"))
    payload_manifest_sha = _file_sha256(
        os.path.join(REQUESTS_DIR, "payload_manifest.csv"))
    cur_cfg = {
        "model": (attempt_params or {}).get("model"),
        "temperature": (attempt_params or {}).get("temperature"),
        "repair_enabled": bool(enable_repair),
        "simulated": bool(simulated),
        # pre-run revision 8 (PENDING author confirmation): the thinking mode
        # is a scoring-relevant transport parameter (thinking ON silently
        # ignores temperature), so it is frozen with the other config keys.
        "thinking_mode": (attempt_params or {}).get("thinking_mode", "default"),
    }
    if os.path.exists(manifest_path):
        stored = json.load(open(manifest_path, "r", encoding="utf-8"))
        # refuse mixing model / temperature / repair / simulated / thinking mode
        for key in ("model", "temperature", "repair_enabled", "simulated",
                    "thinking_mode"):
            if stored.get(key) != cur_cfg[key]:
                raise RuntimeError(
                    "REFUSING to mix runs: %s mismatch (manifest=%r, now=%r). "
                    "Move/delete %s to start a new run."
                    % (key, stored.get(key), cur_cfg[key], out_dir))
        if stored.get("queue_sha256") != queue_sha or \
                stored.get("payload_manifest_sha256") != payload_manifest_sha:
            raise RuntimeError(
                "REFUSING to mix runs: requests/ has changed since the run "
                "started (queue or payload manifest hash mismatch).")
        if (attempt_params or {}).get("timeout_s") != stored.get("timeout_s") or \
                (attempt_params or {}).get("sdk_max_retries") != stored.get("sdk_max_retries"):
            print("  NOTE: timeout/sdk_max_retries differ from the original "
                  "manifest; continuing (they are not scoring parameters).")
    else:
        if resume and os.path.exists(checkpoint_path) \
                and os.path.getsize(checkpoint_path) > 0:
            raise RuntimeError(
                "REFUSING to resume: %s has checkpoint rows but no "
                "run_manifest.json (cannot verify the run configuration). "
                "Move/delete %s to start over." % (out_dir, out_dir))
        save_json(manifest_path, {
            **cur_cfg,
            "timeout_s": (attempt_params or {}).get("timeout_s"),
            "sdk_max_retries": (attempt_params or {}).get("sdk_max_retries"),
            "max_tokens": (attempt_params or {}).get("max_tokens"),
            "queue_sha256": queue_sha,
            "payload_manifest_sha256": payload_manifest_sha,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_revisions": [
                "repair off is the main-experiment default",
                "strict raw-output validation (no default-0, no silent clamp)",
                "structural gate for B responses",
                "explicit timeout + SDK max_retries=0, outer retry with real backoff",
                "uncounted JSON-mode fallback request forbidden",
                "send layer records full request/response/slot_id/attempt_id",
                "per-attempt _last_meta hygiene in the instrumented provider",
                "thinking_mode frozen in the manifest (revision 8, author-approved 2026-09-23)",
            ],
        })

    # verify every frozen request payload BEFORE anything can be sent
    _verify_frozen_requests(queue)

    # Budget guard (run management only): reconstruct its state from the
    # existing attempts log BEFORE anything can be sent (resume-safe), so
    # the cap is not reset by a resume. Recording hooks below keep it live.
    if budget_guard is not None:
        budget_guard.set_queue(queue)
        budget_guard.replay(attempts_path)

    if max_slots:
        queue = queue[:max_slots]

    os.makedirs(out_dir, exist_ok=True)
    results_f = open(results_path, "a", encoding="utf-8")
    check_f = open(checkpoint_path, "a", encoding="utf-8")
    att_f = open(attempts_path, "a", encoding="utf-8")

    def record_attempt(rec):
        att_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        att_f.flush()
        if budget_guard is not None:
            budget_guard.record_attempt(rec)

    # B slots go through the structural gate; A slots use the provider
    # directly. Both share the same retry policy, attempt log and params.
    pre_send = budget_guard.check_before_send if budget_guard is not None else None
    retrying_direct = RetryingProvider(provider, sleep_fn=sleep_fn,
                                       record_attempt=record_attempt,
                                       pre_send_check=pre_send)
    retrying_rulers = RetryingProvider(StructuralGateProvider(provider),
                                       sleep_fn=sleep_fn,
                                       record_attempt=record_attempt,
                                       pre_send_check=pre_send)

    executed = 0
    skipped = 0
    try:
        for i, slot in enumerate(queue):
            if slot["slot_id"] in done:
                skipped += 1
                continue
            if slot_hook is not None:
                slot_hook(slot)
            # per-slot send context: slot id, frozen payload hash, parameters
            for w in (retrying_direct, retrying_rulers):
                w.current_slot_id = slot["slot_id"]
                w.current_expected_sha256 = slot["payload_sha256"]
                w.current_params = attempt_params or {}
            if slot["method"] == "direct":
                messages, _psha = load_payload("direct", slot["opaque_id"])
                out = run_slot_direct(messages, retrying_direct,
                                      slot["text_sha256"])
                out["simulated"] = simulated
            else:
                text = texts[slot["opaque_id"]]
                out = run_slot_rulers(text, retrying_rulers, enable_repair)
                out["simulated"] = simulated
            row = assemble_slot_row(slot, out)
            line = json.dumps(row, ensure_ascii=False) + "\n"
            results_f.write(line)
            check_f.write(line)
            results_f.flush()
            check_f.flush()
            executed += 1
            if not quiet and (i + 1) % 50 == 0:
                print("  progress: %d/%d slots (executed %d, skipped %d)"
                      % (i + 1, len(queue), executed, skipped))
    finally:
        results_f.close()
        check_f.close()
        att_f.close()

    # idempotency: reload results, assert one row per executed slot
    all_rows = {}
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                assert r["slot_id"] not in all_rows, "duplicate slot: %s" % r["slot_id"]
                all_rows[r["slot_id"]] = r
    total_attempts = retrying_direct.total_attempts + retrying_rulers.total_attempts
    return queue, all_rows, executed, skipped, total_attempts


# ---------------------------------------------------------------------------
# Provider instrumentation (live mode only; no network on import)
# ---------------------------------------------------------------------------

def _build_instrumented_class():
    """
    Build a DeepSeekProvider subclass with protocol-revision transport
    settings (additive; the frozen provider class is untouched):
      - explicit per-attempt HTTP timeout (REQUEST_TIMEOUT_S, default 120 s);
      - SDK-level retries OFF (max_retries=0): the outer RetryingProvider owns
        ALL retries, so no attempt can happen uncounted;
      - auto_fallback_json_mode=False: the frozen class's automatic second
        request after a JSON-mode parameter error is FORBIDDEN. A json-mode
        error becomes a counted, non-retryable failed attempt (classified
        API_ERROR) and is recorded like every other attempt.
      - per-attempt metadata hygiene: _last_meta is cleared at the start of
        EVERY generate() call so a failed call can never inherit the previous
        attempt's usage / response id / response_model / finish_reason; a
        call that DID receive a response but failed JSON parsing keeps that
        response's own metadata.
      - thinking mode (pre-run revision 8, APPROVED by the author
        2026-09-23, approval_record.md): thinking_mode="default" sends NO
        thinking parameter (the historical frozen behavior; the API default
        is thinking ON, which silently ignores temperature).
        thinking_mode="disabled" adds the official extra_body={"thinking":
        {"type": "disabled"}} so temperature becomes effective - the
        approved live configuration. The payload hash check is unaffected
        (it hashes the messages only, never the transport parameters).
    """
    from mci_provider import DeepSeekProvider

    class _Instrumented(DeepSeekProvider):
        def __init__(self, timeout_s=REQUEST_TIMEOUT_S,
                     thinking_mode="default", **kwargs):
            kwargs.setdefault("auto_fallback_json_mode", False)
            super(_Instrumented, self).__init__(**kwargs)
            self._timeout_s = timeout_s
            self._thinking_mode = thinking_mode
            # Replace the frozen client (OpenAI(api_key, base_url) with SDK
            # defaults) with the explicit-timeout / no-SDK-retries client.
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url,
                max_retries=SDK_MAX_RETRIES, timeout=self._timeout_s,
            )
            self._last_meta = {}

        def _call_api(self, messages, response_format=None):
            kwargs = dict(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            if self._thinking_mode == "disabled":
                # Official SDK usage (api-docs.deepseek.com/guides/thinking_mode)
                kwargs["extra_body"] = THINKING_EXTRA_BODY
            # No fallback branch here: json-mode errors propagate and are
            # classified by BaseProvider.generate -> counted, not retried.
            resp = self._client.chat.completions.create(**kwargs)
            try:
                meta = {
                    "finish_reason": getattr(resp.choices[0], "finish_reason", None),
                    "created": getattr(resp, "created", None),
                    "id": getattr(resp, "id", None),
                    "response_model": getattr(resp, "model", None),
                }
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    meta["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
            except Exception:
                meta = {}
            self._last_meta = meta
            return {
                "content": resp.choices[0].message.content or "",
                "model": resp.model,
                "simulated_variation": False,
                "error": None,
            }

        def generate(self, messages, response_format=None):
            # Metadata hygiene (protocol revision 7): clear _last_meta before
            # EVERY attempt so a failed call can never inherit the previous
            # attempt's usage / response id / response_model / finish_reason
            # (a failed call's usage stays UNKNOWN - absent, never zero, never
            # the previous call's numbers). A call that DID receive a response
            # but failed JSON parsing keeps its own metadata: _call_api set
            # _last_meta for THIS response before parsing happened.
            self._last_meta = {}
            result = super(_Instrumented, self).generate(messages, response_format)
            if self._last_meta:
                result.update(self._last_meta)
            return result

    return _Instrumented


_InstrumentedClass = None


def make_live_provider(temperature, model_override=None,
                       timeout_s=REQUEST_TIMEOUT_S, thinking_mode="default"):
    """
    Build the live instrumented DeepSeek provider (run_experiment.py --live only).
    Mirrors stability_experiment.py:540-549 (frozen kwargs: temperature,
    max_tokens=16384, model from resolve_provider_config / DEEPSEEK_MODEL),
    plus the protocol-revision transport settings (explicit timeout, SDK
    retries off, JSON-mode fallback forbidden).

    thinking_mode: "default" sends no thinking parameter (historical frozen
    behavior); "disabled" sends the official extra_body
    {"thinking": {"type": "disabled"}} (pre-run revision 8, PENDING author
    confirmation; only then does temperature actually take effect).
    """
    global _InstrumentedClass
    from mci_config import resolve_api_key, resolve_provider_config

    if _InstrumentedClass is None:
        _InstrumentedClass = _build_instrumented_class()

    key_source, api_key = resolve_api_key("deepseek")
    if not api_key:
        raise RuntimeError("DEEPSEEK API key not found (resolve_api_key).")
    cfg = resolve_provider_config("deepseek")
    model = model_override or cfg["model"]
    provider = _InstrumentedClass(
        api_key=api_key,
        model=model,
        base_url=cfg["base_url"],
        temperature=temperature,
        max_tokens=16384,
        timeout_s=timeout_s,
        thinking_mode=thinking_mode,
    )
    return provider, {
        "provider": "deepseek",
        "model": model,
        "base_url": cfg["base_url"],
        "temperature": temperature,
        "max_tokens": 16384,
        "timeout_s": timeout_s,
        "sdk_max_retries": SDK_MAX_RETRIES,
        "thinking_mode": thinking_mode,
        "api_key_source": key_source,   # source description only, never the key
    }
