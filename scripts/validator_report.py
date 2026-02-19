# scripts/validator_report.py
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple

RESULTS_DIR = "results"

def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_seq(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    seq = plan.get("sequence", [])
    return seq if isinstance(seq, list) else []

def _vec(step: Dict[str, Any]) -> Dict[str, Any]:
    v = step.get("vectorization")
    return v if isinstance(v, dict) else {}

def _list6(x: Any) -> List[float]:
    if isinstance(x, list) and len(x) == 6:
        return [float(v) for v in x]
    return [0.0]*6

def compare(raw: Dict[str, Any], val: Dict[str, Any]) -> Dict[str, Any]:
    rseq = _get_seq(raw)
    vseq = _get_seq(val)
    n = min(len(rseq), len(vseq))

    vm_fixed_steps = 0
    frame_changed_steps = 0
    v_changed_count = 0
    m_changed_count = 0

    for i in range(n):
        rv = _vec(rseq[i])
        vv = _vec(vseq[i])

        rframe = (rv.get("frame_mode") or "").upper()
        vframe = (vv.get("frame_mode") or "").upper()
        if rframe and vframe and rframe != vframe:
            frame_changed_steps += 1

        rV = _list6(rv.get("V"))
        rM = _list6(rv.get("M"))
        vV = _list6(vv.get("V"))
        vM = _list6(vv.get("M"))

        # count index-level changes
        v_changed_count += sum(1 for k in range(6) if abs(rV[k] - vV[k]) > 1e-9)
        m_changed_count += sum(1 for k in range(6) if abs(rM[k] - vM[k]) > 1e-9)

        # detect VM-rule fix pattern: raw had both nonzero at same index but val zeroed M
        for k in range(6):
            if abs(rV[k]) > 1e-9 and abs(rM[k]) > 1e-9 and abs(vM[k]) < 1e-12:
                vm_fixed_steps += 1
                break

    return {
        "steps_raw": len(rseq),
        "steps_valid": len(vseq),
        "vm_rule_fixed_steps": vm_fixed_steps,
        "frame_changed_steps": frame_changed_steps,
        "V_index_changes": v_changed_count,
        "M_index_changes": m_changed_count,
    }

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    # minimal CSV writer (no pandas dependency)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

def to_markdown(row: Dict[str, Any]) -> str:
    keys = list(row.keys())
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"]*len(keys)) + " |"
    vals = "| " + " | ".join(str(row[k]) for k in keys) + " |"
    return "\n".join([header, sep, vals])

def main():
    task = input("Task Name (exact): ").strip()
    raw_path = os.path.join(RESULTS_DIR, f"{task}__raw.json")
    val_path = os.path.join(RESULTS_DIR, f"{task}.json")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing: {raw_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing: {val_path}")

    raw = _load(raw_path)
    val = _load(val_path)

    summary = compare(raw, val)
    summary["task"] = task

    # Save CSV (table-1-ready)
    out_csv = os.path.join(RESULTS_DIR, f"{task}__validator_summary.csv")
    write_csv(out_csv, [summary])

    # Also print markdown for quick paste into slides/notes
    print("\n=== Validator Before/After Summary (paste-ready) ===")
    print(to_markdown(summary))
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
