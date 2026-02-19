# scripts/validator.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from __future__ import annotations

ALLOWED_FRAMES = {"WORLD", "CONTACT", "FUNCTIONAL"}
ALLOWED_SUBTASKS = {
    "pre_grasp", "grasp", "move_by_displacement", "rotate", "move_to_pose", "place", "release"
}

DEFAULT_MAX_ABS_V = 3.0
DEFAULT_MAX_ABS_M = 50.0


@dataclass
class ValidationIssue:
    level: str  # "ERROR" | "WARN"
    code: str
    message: str
    path: str = ""


@dataclass
class ValidationResult:
    ok: bool
    sanitized: Dict[str, Any]
    issues: List[ValidationIssue] = field(default_factory=list)

    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "ERROR"]

    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "WARN"]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(float(x))


def _as_list6(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list) or len(x) != 6:
        return None
    out: List[float] = []
    for v in x:
        if not _is_number(v):
            return None
        out.append(float(v))
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_frame(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    f = x.strip().upper()
    return f if f in ALLOWED_FRAMES else None


def _norm_subtask(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    return x.strip().lower().replace(" ", "_")


def build_id_set(points_info_by_object: Dict[str, Dict[str, Any]]) -> Dict[str, Set[int]]:
    """
    Returns a merged id set per point kind:
      {"contact": {0,1,...}, "functional": {0,1,...}}
    Works with points_info.json where each entry may contain id: int or id: [ints].
    """
    contact: Set[int] = set()
    functional: Set[int] = set()

    for _, info in points_info_by_object.items():
        if not isinstance(info, dict):
            continue
        for entry in info.get("contact_points", []):
            if isinstance(entry, dict) and "id" in entry:
                cid = entry["id"]
                if isinstance(cid, int):
                    contact.add(cid)
                elif isinstance(cid, list):
                    contact.update([x for x in cid if isinstance(x, int)])

        for entry in info.get("functional_points", []):
            if isinstance(entry, dict) and "id" in entry:
                fid = entry["id"]
                if isinstance(fid, int):
                    functional.add(fid)
                elif isinstance(fid, list):
                    functional.update([x for x in fid if isinstance(x, int)])

    return {"contact": contact, "functional": functional}


def validate_plan(
    plan: Dict[str, Any],
    *,
    id_sets: Optional[Dict[str, Set[int]]] = None,
    auto_fix: bool = True,
    max_abs_v: float = DEFAULT_MAX_ABS_V,
    max_abs_m: float = DEFAULT_MAX_ABS_M,
) -> ValidationResult:
    """
    Validates current planner output format:

    {
      "task": "...",
      "sequence": [
        {
          "subtask": "...",
          "frame": "WORLD|CONTACT|FUNCTIONAL",
          "actor_point": int|null,
          "target_point": int|null,
          "V": [6],
          "M": [6],
          "notes": "..."
        }
      ]
    }
    """
    issues: List[ValidationIssue] = []
    sanitized = json.loads(json.dumps(plan))  # deep copy

    seq = sanitized.get("sequence")
    if not isinstance(seq, list):
        issues.append(ValidationIssue("ERROR", "NO_SEQUENCE", "Top-level 'sequence' must be a list.", "sequence"))
        return ValidationResult(ok=False, sanitized=sanitized, issues=issues)

    has_error = False

    if len(seq) < 1:
        issues.append(ValidationIssue("ERROR", "EMPTY_SEQUENCE", "Sequence must contain at least 1 step.", "sequence"))
        return ValidationResult(ok=False, sanitized=sanitized, issues=issues)

    # soft: keep sequence short
    if len(seq) > 8:
        issues.append(ValidationIssue("WARN", "TOO_MANY_STEPS", f"Sequence has {len(seq)} steps; recommended <= 8.", "sequence"))

    for i, step in enumerate(seq):
        p = f"sequence[{i}]"
        if not isinstance(step, dict):
            issues.append(ValidationIssue("ERROR", "STEP_NOT_OBJECT", "Each step must be an object.", p))
            has_error = True
            continue

        subtask = _norm_subtask(step.get("subtask"))
        if subtask is None:
            issues.append(ValidationIssue("ERROR", "BAD_SUBTASK", "Missing/invalid 'subtask'.", f"{p}.subtask"))
            has_error = True
        else:
            if auto_fix:
                step["subtask"] = subtask
            if subtask not in ALLOWED_SUBTASKS:
                issues.append(ValidationIssue("WARN", "UNKNOWN_SUBTASK", f"Subtask '{subtask}' not in allowed set (will still validate).", f"{p}.subtask"))

        frame = _norm_frame(step.get("frame"))
        if frame is None:
            issues.append(ValidationIssue("ERROR", "BAD_FRAME", f"'frame' must be one of {sorted(ALLOWED_FRAMES)}.", f"{p}.frame"))
            has_error = True
        else:
            if auto_fix:
                step["frame"] = frame

        V = _as_list6(step.get("V"))
        M = _as_list6(step.get("M"))
        if V is None:
            issues.append(ValidationIssue("ERROR", "BAD_V", "'V' must be a list of 6 numbers.", f"{p}.V"))
            has_error = True
        if M is None:
            issues.append(ValidationIssue("ERROR", "BAD_M", "'M' must be a list of 6 numbers.", f"{p}.M"))
            has_error = True

        # validate points (int or null)
        for k in ("actor_point", "target_point"):
            if k not in step:
                issues.append(ValidationIssue("WARN", "MISSING_POINT_KEY", f"Missing '{k}' (allowed to be null).", f"{p}.{k}"))
                continue

            val = step.get(k)
            if val is None:
                continue
            if not isinstance(val, int):
                issues.append(ValidationIssue("ERROR", "POINT_NOT_INT", f"'{k}' must be int or null.", f"{p}.{k}"))
                has_error = True
                continue

            # optional: check within id set (best-effort)
            if id_sets:
                # we cannot know whether this id is contact or functional here;
                # treat as "must exist in either" to avoid false negatives.
                all_ids = (id_sets.get("contact", set()) | id_sets.get("functional", set()))
                if all_ids and val not in all_ids:
                    issues.append(ValidationIssue("WARN", "POINT_ID_NOT_FOUND",
                        f"'{k}' id={val} not found in merged points_info ids.", f"{p}.{k}"))

        # enforce V/M rules and clamp
        if V is not None and M is not None:
            if auto_fix:
                V = [_clamp(v, -max_abs_v, max_abs_v) for v in V]
                M = [_clamp(m, -max_abs_m, max_abs_m) for m in M]

            violated = [k for k in range(6) if abs(V[k]) > 1e-9 and abs(M[k]) > 1e-9]
            if violated:
                if auto_fix:
                    for k in violated:
                        M[k] = 0.0
                    issues.append(ValidationIssue("WARN", "VM_RULE_FIXED",
                        f"Auto-fixed: set M[{violated}] to 0 because V was non-zero.", p))
                else:
                    issues.append(ValidationIssue("ERROR", "VM_RULE_VIOLATION",
                        f"Rule violated at indices {violated}.", p))
                    has_error = True

            if auto_fix:
                step["V"] = V
                step["M"] = M

        # soft: encourage sparse V
        if V is not None:
            nz = sum(1 for x in V if abs(x) > 1e-9)
            if nz == 0:
                issues.append(ValidationIssue("WARN", "ZERO_TWIST", "V is all zeros. Is this step necessary?", p))
            if nz > 2:
                issues.append(ValidationIssue("WARN", "DENSE_TWIST", f"V has {nz} non-zero components; prefer sparse.", p))

    return ValidationResult(ok=(not has_error), sanitized=sanitized, issues=issues)


def issues_to_text(issues: List[ValidationIssue]) -> str:
    return "\n".join(
        f"[{it.level}] {it.code}" + (f" @ {it.path}: " if it.path else ": ") + it.message
        for it in issues
    )
