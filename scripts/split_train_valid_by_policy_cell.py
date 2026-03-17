from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# Config (edit here first)
# =========================
LIFE_PERFORMANCE_CSV = Path("data/processed/life_performance.csv")
POLICY_MEANING_CSV = Path("data/processed/policy_meaning.csv")

TRAIN_OUTPUT_CSV = Path("data/processed/train_policy_cell_samples.csv")
VALID_OUTPUT_CSV = Path("data/processed/valid_policy_cell_samples.csv")

ENCODING = "utf-8"
VALID_RATIO_MULTI_POLICY = 0.20
VALID_RATIO_SINGLE_POLICY = 0.10
FORCE_INCLUDE_ALL_VARCHARGE_IN_VALID = True


def load_policy_meaning(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing policy meaning file: {path}")

    mapping: Dict[str, dict] = {}
    with path.open("r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        required = {"policy", "initial_c_rate", "switch_soc_percent", "post_switch_c_rate"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            policy = row["policy"]
            mapping[policy] = {
                "initial_c_rate": row.get("initial_c_rate", ""),
                "switch_soc_percent": row.get("switch_soc_percent", ""),
                "post_switch_c_rate": row.get("post_switch_c_rate", ""),
                "note": row.get("note", ""),
            }
    return mapping


def load_policy_cell_samples(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing life performance file: {path}")

    max_cycles: Dict[Tuple[str, str], int] = {}
    with path.open("r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        required = {"policy", "cell_code", "cycles"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            try:
                policy = row["policy"]
                cell_code = row["cell_code"]
                cycles = int(float(row["cycles"]))
            except (TypeError, ValueError, KeyError):
                continue

            key = (policy, cell_code)
            old = max_cycles.get(key)
            if old is None or cycles > old:
                max_cycles[key] = cycles

    samples = [
        {"policy": p, "cell_code": c, "max_cycles": mc}
        for (p, c), mc in max_cycles.items()
    ]
    return samples


def pick_evenly_spread(items: List[dict], k: int) -> List[dict]:
    if k <= 0 or not items:
        return []
    sorted_items = sorted(items, key=lambda x: (x["max_cycles"], x["cell_code"]))
    n = len(sorted_items)
    if k >= n:
        return sorted_items
    if k == 1:
        return [sorted_items[n // 2]]

    idxs = set()
    for i in range(k):
        idx = round(i * (n - 1) / (k - 1))
        idxs.add(idx)
    picked = [sorted_items[i] for i in sorted(idxs)]

    # In case rounding deduplicates indexes, fill from remaining middle-out.
    if len(picked) < k:
        remaining = [x for i, x in enumerate(sorted_items) if i not in idxs]
        needed = k - len(picked)
        picked.extend(remaining[:needed])
    return picked[:k]


def split_samples(samples: List[dict], policy_meta: Dict[str, dict]) -> Tuple[List[dict], List[dict]]:
    by_policy: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_policy[s["policy"]].append(s)

    valid_keys = set()

    multi_policy_items = {p: ss for p, ss in by_policy.items() if len(ss) > 1}
    single_policy_items = {p: ss for p, ss in by_policy.items() if len(ss) == 1}

    # 1) Multi-cell policies: select part of cells for validation within each policy.
    for policy, cells in sorted(multi_policy_items.items()):
        n = len(cells)
        k = max(1, round(n * VALID_RATIO_MULTI_POLICY))
        chosen = pick_evenly_spread(cells, k)
        for x in chosen:
            valid_keys.add((x["policy"], x["cell_code"]))

    # 2) Include variable-charge policies in validation (coverage priority).
    if FORCE_INCLUDE_ALL_VARCHARGE_IN_VALID:
        for policy, cells in by_policy.items():
            note = policy_meta.get(policy, {}).get("note", "")
            if ("variable_charge" in note) or policy.startswith("VARCHARGE_"):
                for x in cells:
                    valid_keys.add((x["policy"], x["cell_code"]))

    # 3) Single-cell policies: include a subset to keep representation.
    single_items = [ss[0] for _, ss in sorted(single_policy_items.items())]
    target_single_valid = max(1, round(len(single_items) * VALID_RATIO_SINGLE_POLICY)) if single_items else 0
    already_single_valid = sum(
        1 for x in single_items if (x["policy"], x["cell_code"]) in valid_keys
    )
    need_more = max(0, target_single_valid - already_single_valid)
    if need_more > 0:
        remaining_single = [x for x in single_items if (x["policy"], x["cell_code"]) not in valid_keys]
        chosen = pick_evenly_spread(remaining_single, need_more)
        for x in chosen:
            valid_keys.add((x["policy"], x["cell_code"]))

    # Build final splits and attach policy parameters.
    valid_rows: List[dict] = []
    train_rows: List[dict] = []

    for s in samples:
        policy = s["policy"]
        meta = policy_meta.get(policy, {})
        row = {
            "policy": policy,
            "cell_code": s["cell_code"],
            "initial_c_rate": meta.get("initial_c_rate", ""),
            "switch_soc_percent": meta.get("switch_soc_percent", ""),
            "post_switch_c_rate": meta.get("post_switch_c_rate", ""),
            "max_cycles": s["max_cycles"],
            "policy_note": meta.get("note", ""),
        }
        if (s["policy"], s["cell_code"]) in valid_keys:
            valid_rows.append(row)
        else:
            train_rows.append(row)

    sort_key = lambda x: (x["policy"], x["cell_code"])
    train_rows.sort(key=sort_key)
    valid_rows.sort(key=sort_key)
    return train_rows, valid_rows


def save_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "cell_code",
        "initial_c_rate",
        "switch_soc_percent",
        "post_switch_c_rate",
        "max_cycles",
        "policy_note",
    ]
    with path.open("w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    policy_meta = load_policy_meaning(POLICY_MEANING_CSV)
    samples = load_policy_cell_samples(LIFE_PERFORMANCE_CSV)
    train_rows, valid_rows = split_samples(samples, policy_meta)

    save_csv(TRAIN_OUTPUT_CSV, train_rows)
    save_csv(VALID_OUTPUT_CSV, valid_rows)

    print(f"Total policy+cell samples: {len(samples)}")
    print(f"Train samples: {len(train_rows)}")
    print(f"Valid samples: {len(valid_rows)}")
    print(f"Saved train to: {TRAIN_OUTPUT_CSV}")
    print(f"Saved valid to: {VALID_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
