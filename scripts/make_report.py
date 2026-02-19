# scripts/make_reports.py
from __future__ import annotations

import os
import re
import csv
from typing import Any, Dict, List, Tuple

from validator import ValidationResult, issues_to_text


def safe_filename(name: str) -> str:
    # 파일명 안전하게(공백/특수문자 정리)
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-가-힣]+", "", s)
    return s


def _list6(x: Any) -> List[float]:
    if isinstance(x, list) and len(x) == 6:
        out = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out
    return [0.0] * 6


def plan_to_step_rows(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    seq = plan.get("sequence", [])
    if not isinstance(seq, list):
        return []

    rows: List[Dict[str, Any]] = []
    for i, step in enumerate(seq):
        if not isinstance(step, dict):
            continue
        V = _list6(step.get("V"))
        M = _list6(step.get("M"))

        rows.append({
            "idx": i,
            "subtask": step.get("subtask", ""),
            "frame": step.get("frame", ""),
            "actor_obj": step.get("actor_obj", step.get("actor", "")),
            "actor_point": step.get("actor_point", None),
            "target_obj": step.get("target_obj", step.get("target", "")),
            "target_point": step.get("target_point", None),
            "vx": V[0], "vy": V[1], "vz": V[2], "wx": V[3], "wy": V[4], "wz": V[5],
            "mx": M[0], "my": M[1], "mz": M[2], "mrx": M[3], "mry": M[4], "mrz": M[5],
            "notes": step.get("notes", ""),
        })
    return rows


def compare_raw_validated(raw: Dict[str, Any], validated: Dict[str, Any]) -> Dict[str, Any]:
    rseq = raw.get("sequence", [])
    vseq = validated.get("sequence", [])
    rseq = rseq if isinstance(rseq, list) else []
    vseq = vseq if isinstance(vseq, list) else []
    n = min(len(rseq), len(vseq))

    frame_changed_steps = 0
    v_index_changes = 0
    m_index_changes = 0
    point_changed_steps = 0
    subtask_changed_steps = 0

    def _get(step, key, default=None):
        return step.get(key, default) if isinstance(step, dict) else default

    for i in range(n):
        rs = rseq[i] if isinstance(rseq[i], dict) else {}
        vs = vseq[i] if isinstance(vseq[i], dict) else {}

        if str(_get(rs, "frame", "")).upper() != str(_get(vs, "frame", "")).upper():
            frame_changed_steps += 1

        if str(_get(rs, "subtask", "")).lower() != str(_get(vs, "subtask", "")).lower():
            subtask_changed_steps += 1

        rV = _list6(_get(rs, "V", []))
        rM = _list6(_get(rs, "M", []))
        vV = _list6(_get(vs, "V", []))
        vM = _list6(_get(vs, "M", []))

        v_index_changes += sum(1 for k in range(6) if abs(rV[k] - vV[k]) > 1e-9)
        m_index_changes += sum(1 for k in range(6) if abs(rM[k] - vM[k]) > 1e-9)

        if (_get(rs, "actor_point", None), _get(rs, "target_point", None)) != (_get(vs, "actor_point", None), _get(vs, "target_point", None)):
            point_changed_steps += 1

    # frame 분포(발표용)
    frames = {"WORLD": 0, "CONTACT": 0, "FUNCTIONAL": 0}
    for s in vseq:
        if isinstance(s, dict):
            f = str(s.get("frame", "")).upper()
            if f in frames:
                frames[f] += 1

    # WORLD lift step count(간단한 체크)
    world_lift_steps = 0
    for s in vseq:
        if isinstance(s, dict) and str(s.get("frame", "")).upper() == "WORLD":
            V = _list6(s.get("V"))
            if V[2] > 1e-9:
                world_lift_steps += 1

    return {
        "steps_raw": len(rseq),
        "steps_validated": len(vseq),
        "frame_changed_steps": frame_changed_steps,
        "subtask_changed_steps": subtask_changed_steps,
        "V_index_changes": v_index_changes,
        "M_index_changes": m_index_changes,
        "point_changed_steps": point_changed_steps,
        "frames_WORLD": frames["WORLD"],
        "frames_CONTACT": frames["CONTACT"],
        "frames_FUNCTIONAL": frames["FUNCTIONAL"],
        "world_lift_steps": world_lift_steps,
    }


def issue_code_counts(issues) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for it in issues:
        counts[it.code] = counts.get(it.code, 0) + 1
    return counts


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        # 빈 파일도 만들어 둠
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_summary_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_reports(
    task_name: str,
    raw_plan: Dict[str, Any],
    val: ValidationResult,
    output_dir: str = "results",
) -> Dict[str, str]:
    """
    생성 파일:
      results/reports/<task>__steps_raw.csv
      results/reports/<task>__steps_validated.csv
      results/reports/<task>__validator_issues.txt
      results/reports/<task>__validator_summary.csv
      results/reports/summary.csv  (append)
    """
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    slug = safe_filename(task_name)

    raw_steps_csv = os.path.join(reports_dir, f"{slug}__steps_raw.csv")
    val_steps_csv = os.path.join(reports_dir, f"{slug}__steps_validated.csv")
    issues_txt = os.path.join(reports_dir, f"{slug}__validator_issues.txt")
    summary_csv = os.path.join(reports_dir, f"{slug}__validator_summary.csv")
    global_summary_csv = os.path.join(reports_dir, "summary.csv")

    write_csv(raw_steps_csv, plan_to_step_rows(raw_plan))
    write_csv(val_steps_csv, plan_to_step_rows(val.sanitized))

    with open(issues_txt, "w", encoding="utf-8") as f:
        f.write(issues_to_text(val.issues) if val.issues else "[Validator] No issues.\n")

    cmp = compare_raw_validated(raw_plan, val.sanitized)
    code_counts = issue_code_counts(val.issues)

    summary_row = {
        "task": task_name,
        "ok": val.ok,
        "errors": sum(1 for x in val.issues if x.level == "ERROR"),
        "warnings": sum(1 for x in val.issues if x.level == "WARN"),
        **cmp,
        # 자주 보는 fix 카운트들(없으면 0)
        "VM_RULE_FIXED": code_counts.get("VM_RULE_FIXED", 0),
        "FRAME_HARD_FIXED": code_counts.get("FRAME_HARD_FIXED", 0),
        "POINT_PARSED": code_counts.get("POINT_PARSED", 0),
        "ZERO_STEP": code_counts.get("ZERO_STEP", 0),
    }

    write_csv(summary_csv, [summary_row])
    append_summary_csv(global_summary_csv, summary_row)

    return {
        "raw_steps_csv": raw_steps_csv,
        "validated_steps_csv": val_steps_csv,
        "issues_txt": issues_txt,
        "summary_csv": summary_csv,
        "global_summary_csv": global_summary_csv,
    }
