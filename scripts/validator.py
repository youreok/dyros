# scripts/validator.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_FRAMES = {"WORLD", "CONTACT", "FUNCTIONAL"}

# 너가 pre_grasp를 제외했다고 했으니 기본 허용 목록은 이렇게.
# 다만 LLM이 실수로 다른 subtask를 내도 바로 터지지 않게 기본은 WARN 처리로 설계.
ALLOWED_SUBTASKS = {
    "grasp",
    "move_by_displacement",
    "move_to_pose",
    "rotate",
    "place",
    "release",
}

# frame을 강제할지 여부(발표 안정성↑). 기본은 WARN+auto-fix로 두는 걸 추천.
HARD_FRAME_BY_SUBTASK = {
    "grasp": "CONTACT",
    "rotate": "FUNCTIONAL",
}


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


def issues_to_text(issues: List[ValidationIssue]) -> str:
    lines = []
    for it in issues:
        loc = f" @ {it.path}" if it.path else ""
        lines.append(f"[{it.level}] {it.code}{loc}: {it.message}")
    return "\n".join(lines)


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


def _try_parse_point_id(v: Any) -> Optional[int]:
    """
    허용: int, "0", "contact_point_0", "functional_point_2" 같은 문자열
    """
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.isdigit():
            return int(s)
        # contact_point_0 형태
        for prefix in ("contact_point_", "functional_point_", "point_"):
            if s.startswith(prefix):
                tail = s[len(prefix):]
                if tail.isdigit():
                    return int(tail)
    return None


def build_point_id_index(points_info_by_object: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Set[int]]]:
    """
    returns:
    {
      "<obj>": {
        "contact_point": set(int),
        "functional_point": set(int),
        "any_point": set(int)
      },
      "_union": {...}
    }
    """
    out: Dict[str, Dict[str, Set[int]]] = {}
    union_contact: Set[int] = set()
    union_functional: Set[int] = set()

    for obj, info in points_info_by_object.items():
        contact_ids: Set[int] = set()
        functional_ids: Set[int] = set()

        if isinstance(info, dict):
            for entry in info.get("contact_points", []):
                if isinstance(entry, dict) and "id" in entry:
                    cid = entry["id"]
                    if isinstance(cid, int):
                        contact_ids.add(cid)
                    elif isinstance(cid, list):
                        contact_ids.update([x for x in cid if isinstance(x, int)])

            for entry in info.get("functional_points", []):
                if isinstance(entry, dict) and "id" in entry:
                    fid = entry["id"]
                    if isinstance(fid, int):
                        functional_ids.add(fid)
                    elif isinstance(fid, list):
                        functional_ids.update([x for x in fid if isinstance(x, int)])

        union_contact |= contact_ids
        union_functional |= functional_ids

        out[obj] = {
            "contact_point": contact_ids,
            "functional_point": functional_ids,
            "any_point": (contact_ids | functional_ids),
        }

    out["_union"] = {
        "contact_point": union_contact,
        "functional_point": union_functional,
        "any_point": (union_contact | union_functional),
    }
    return out


def validate_plan(
    plan: Dict[str, Any],
    point_index: Dict[str, Dict[str, Set[int]]],
    *,
    auto_fix: bool = True,
    strict_subtasks: bool = False,
    max_abs_v: float = 3.0,
    max_abs_m: float = 50.0,
) -> ValidationResult:
    """
    기대 포맷(너의 현재 prompt 출력과 맞춤):

    {
      "task": "...",
      "sequence": [
        {
          "subtask": "...",
          "frame": "WORLD|CONTACT|FUNCTIONAL",
          "actor_obj": "wrench" or null (optional but recommended),
          "target_obj": "bolt"  or null (optional but recommended),
          "actor_point": int|null,
          "target_point": int|null,
          "V": [6],
          "M": [6],
          "notes": "..."
        }, ...
      ]
    }
    """
    issues: List[ValidationIssue] = []
    sanitized = json.loads(json.dumps(plan))  # deep copy

    # --- top-level checks
    if "task" not in sanitized or not isinstance(sanitized.get("task"), str) or not sanitized["task"].strip():
        issues.append(ValidationIssue("WARN", "MISSING_TASK", "Top-level 'task' is missing/invalid (recommended).", "task"))

    seq = sanitized.get("sequence")
    if not isinstance(seq, list):
        issues.append(ValidationIssue("ERROR", "NO_SEQUENCE", "Top-level 'sequence' must be a list.", "sequence"))
        return ValidationResult(ok=False, sanitized=sanitized, issues=issues)

    if len(seq) == 0:
        issues.append(ValidationIssue("ERROR", "EMPTY_SEQUENCE", "Sequence must contain at least one step.", "sequence"))
        return ValidationResult(ok=False, sanitized=sanitized, issues=issues)

    # 발표용 권장: 너무 길면 경고
    if len(seq) > 8:
        issues.append(ValidationIssue("WARN", "TOO_MANY_STEPS", f"Sequence has {len(seq)} steps; recommended <= 8.", "sequence"))

    union_contact = point_index.get("_union", {}).get("contact_point", set())
    union_functional = point_index.get("_union", {}).get("functional_point", set())
    union_any = point_index.get("_union", {}).get("any_point", set())

    has_error = False

    for i, step in enumerate(seq):
        p = f"sequence[{i}]"
        if not isinstance(step, dict):
            issues.append(ValidationIssue("ERROR", "STEP_NOT_OBJECT", "Each step must be an object.", p))
            has_error = True
            continue

        # --- normalize subtask
        subtask_raw = step.get("subtask")
        subtask = _norm_subtask(subtask_raw)
        if subtask is None:
            issues.append(ValidationIssue("ERROR", "BAD_SUBTASK", "Missing/invalid 'subtask'.", f"{p}.subtask"))
            has_error = True
        else:
            if auto_fix:
                step["subtask"] = subtask
            if subtask not in ALLOWED_SUBTASKS:
                if strict_subtasks:
                    issues.append(ValidationIssue("ERROR", "SUBTASK_NOT_ALLOWED", f"Subtask '{subtask}' not allowed.", f"{p}.subtask"))
                    has_error = True
                else:
                    issues.append(ValidationIssue("WARN", "UNKNOWN_SUBTASK", f"Subtask '{subtask}' not in allowed set (will continue).", f"{p}.subtask"))

        # --- normalize frame
        frame_raw = step.get("frame")
        frame = _norm_frame(frame_raw)
        if frame is None:
            issues.append(ValidationIssue("ERROR", "BAD_FRAME", f"'frame' must be one of {sorted(ALLOWED_FRAMES)}.", f"{p}.frame"))
            has_error = True
        else:
            if auto_fix:
                step["frame"] = frame

        # --- optional: actor_obj / target_obj
        actor_obj = step.get("actor_obj", step.get("actor", None))
        target_obj = step.get("target_obj", step.get("target", None))
        # keep as-is but validate if present
        if actor_obj is not None and not isinstance(actor_obj, str):
            issues.append(ValidationIssue("WARN", "BAD_ACTOR_OBJ", "'actor_obj' should be a string or null.", f"{p}.actor_obj"))
        if target_obj is not None and not isinstance(target_obj, str):
            issues.append(ValidationIssue("WARN", "BAD_TARGET_OBJ", "'target_obj' should be a string or null.", f"{p}.target_obj"))

        # --- actor_point / target_point (int|null; also parse some strings)
        for k in ("actor_point", "target_point"):
            if k not in step:
                issues.append(ValidationIssue("WARN", "MISSING_POINT_KEY", f"Missing '{k}' (allowed to be null).", f"{p}.{k}"))
                continue
            parsed = _try_parse_point_id(step.get(k))
            if step.get(k) is None:
                continue
            if parsed is None:
                issues.append(ValidationIssue("ERROR", "POINT_NOT_INT", f"'{k}' must be int or null.", f"{p}.{k}"))
                has_error = True
            else:
                if auto_fix and parsed != step.get(k):
                    step[k] = parsed
                    issues.append(ValidationIssue("WARN", "POINT_PARSED", f"Parsed '{k}' string -> int ({parsed}).", f"{p}.{k}"))

        # --- V / M
        V = _as_list6(step.get("V"))
        M = _as_list6(step.get("M"))
        if V is None:
            issues.append(ValidationIssue("ERROR", "BAD_V", "'V' must be a list of 6 numbers.", f"{p}.V"))
            has_error = True
        if M is None:
            issues.append(ValidationIssue("ERROR", "BAD_M", "'M' must be a list of 6 numbers.", f"{p}.M"))
            has_error = True

        if V is not None and M is not None:
            # clamp
            if auto_fix:
                V = [_clamp(v, -max_abs_v, max_abs_v) for v in V]
                M = [_clamp(m, -max_abs_m, max_abs_m) for m in M]

            # V[i]!=0 -> M[i]==0
            violated = [k for k in range(6) if abs(V[k]) > 1e-9 and abs(M[k]) > 1e-9]
            if violated:
                if auto_fix:
                    for k in violated:
                        M[k] = 0.0
                    issues.append(ValidationIssue("WARN", "VM_RULE_FIXED", f"Auto-fixed: zeroed M at indices {violated}.", p))
                else:
                    issues.append(ValidationIssue("ERROR", "VM_RULE_VIOLATION", f"Rule violated at indices {violated}.", p))
                    has_error = True

            if auto_fix:
                step["V"] = V
                step["M"] = M

            # 발표용 경고(불필요한 step 찾기 쉬움)
            nz = sum(1 for x in V if abs(x) > 1e-9)
            if nz == 0 and sum(1 for x in M if abs(x) > 1e-9) == 0:
                issues.append(ValidationIssue("WARN", "ZERO_STEP", "V and M are all zeros (step may be redundant).", p))
            elif nz > 2:
                issues.append(ValidationIssue("WARN", "DENSE_TWIST", f"V has {nz} non-zero components; prefer sparse.", p))

        # --- hard frame rules (발표 안정성↑)
        if subtask in HARD_FRAME_BY_SUBTASK and frame is not None:
            required = HARD_FRAME_BY_SUBTASK[subtask]
            if frame != required:
                if auto_fix:
                    step["frame"] = required
                    issues.append(ValidationIssue("WARN", "FRAME_HARD_FIXED", f"Auto-fixed frame: {frame} -> {required} for '{subtask}'.", f"{p}.frame"))
                else:
                    issues.append(ValidationIssue("ERROR", "FRAME_HARD_VIOLATION", f"Subtask '{subtask}' requires frame '{required}'.", f"{p}.frame"))
                    has_error = True

        
        # --- after V/M are validated, clamped, and VM rule applied
        all_zero = (
            V is not None and M is not None and
            all(abs(x) < 1e-12 for x in V) and
            all(abs(x) < 1e-12 for x in M)
        )

        if all_zero and subtask in {"move_to_pose", "place", "move_by_displacement"}:
            if auto_fix:
                # default approach axis policy
                if frame in {"FUNCTIONAL", "CONTACT", "WORLD"}:
                    V[2] = 1.0  # +z of the chosen frame
                    step["V"] = V
                    step["M"] = M
                    issues.append(ValidationIssue(
                        "WARN",
                        "ZERO_STEP_FILLED",
                        f"Filled all-zero step with default approach Vz=+1.0 in frame={frame}",
                        p
                    ))
            else:
                issues.append(ValidationIssue(
                    "ERROR",
                    "ZERO_STEP_NOT_ALLOWED",
                    "All-zero V/M not allowed for this subtask.",
                    p
                ))
                has_error = True



        # --- point id validity check (frame 기반으로 contact/functional 구분)
        # frame=CONTACT => contact_point id여야 함
        # frame=FUNCTIONAL => functional_point id여야 함
        # frame=WORLD => 어떤 id든 가능(하지만 있으면 union_any 안에 있어야 함)
        expected_kind: Optional[str] = None
        if frame == "CONTACT":
            expected_kind = "contact_point"
        elif frame == "FUNCTIONAL":
            expected_kind = "functional_point"

        def _check_id(field: str, obj_name: Optional[str]) -> None:
            nonlocal has_error
            pid = step.get(field)
            if pid is None or not isinstance(pid, int):
                return

            if frame == "WORLD":
                # WORLD는 타입 강제 안 함(그래도 id가 존재하는지는 union_any로 체크)
                if union_any and pid not in union_any:
                    issues.append(ValidationIssue("WARN", "POINT_ID_NOT_FOUND",
                        f"{field}={pid} not found in any points_info id.", f"{p}.{field}"))
                return

            if expected_kind is None:
                return

            # object가 명시돼 있으면 해당 object의 해당 kind에서 검사
            if isinstance(obj_name, str) and obj_name in point_index:
                allowed = point_index[obj_name].get(expected_kind, set())
                if allowed and pid not in allowed:
                    issues.append(ValidationIssue("ERROR", "POINT_ID_INVALID_FOR_OBJECT",
                        f"{field}={pid} not in {obj_name}.{expected_kind} ids.", f"{p}.{field}"))
                    has_error = True
                return

            # object 미명시 => union set에서 검사
            union_set = union_contact if expected_kind == "contact_point" else union_functional
            if union_set and pid not in union_set:
                issues.append(ValidationIssue("ERROR", "POINT_ID_INVALID_FOR_FRAME",
                    f"{field}={pid} not valid for frame={frame} (expected {expected_kind}).", f"{p}.{field}"))
                has_error = True

        _check_id("actor_point", actor_obj if isinstance(actor_obj, str) else None)
        _check_id("target_point", target_obj if isinstance(target_obj, str) else None)

    return ValidationResult(ok=(not has_error), sanitized=sanitized, issues=issues)
