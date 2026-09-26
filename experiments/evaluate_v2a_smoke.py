"""
Smoke evaluator for blocking_v2a_candidates.tsv.

- Ground truth is aligned by source1_entity_id (GT file order != S1 order).
- Empty candidate lists are read as "" (keep_default_na=False), never "nan".
- Target IDs are classified as S2/S3 using the COMPLETE S2/S3 ID sets.
- Because the smoke blocker only loads the first SMOKE_ROWS rows of
  S2/S3, metrics are reported twice:
    ALL GT       - against every true target (understates smoke recall)
    REACHABLE    - true targets restricted to those the smoke run loaded
"""
from pathlib import Path
import os
import pandas as pd

N = int(os.environ.get("SMOKE_ROWS", "200000"))

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT.parent / "student_resource"
TRAIN = RESOURCE / "dataset" / "train"

GT_PATH = TRAIN / "train_ground_truth.tsv"
S1_PATH = TRAIN / "train_source1.tsv"
S2_PATH = TRAIN / "train_source2.tsv"
S3_PATH = TRAIN / "train_source3.tsv"
CAND_PATH = Path(os.environ.get(
    "CAND_PATH", ROOT / "experiments" / "blocking_v2a_candidates.tsv"
))

READ = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def read_ids(path, nrows=None):
    return pd.read_csv(path, usecols=["entity_id"], nrows=nrows, **READ)[
        "entity_id"
    ]


def parse_ids(x):
    return {v.strip() for v in x.split(",") if v.strip()}


def f05(tp, n_pred, n_true):
    """Candidate-set F0.5 for one S1 row."""
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    p = tp / n_pred
    r = tp / n_true
    return 1.25 * p * r / (0.25 * p + r)


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
s1_ids = read_ids(S1_PATH, N)

cand = pd.read_csv(CAND_PATH, **READ)
gt = pd.read_csv(
    GT_PATH, usecols=["source1_entity_id", "matched_entity_ids"], **READ
)

print("Loading complete S2/S3 ID sets...")
s2_all_series = read_ids(S2_PATH)
s3_all_series = read_ids(S3_PATH)
s2_all = set(s2_all_series)
s3_all = set(s3_all_series)
# What the smoke blocker actually loaded (first N rows of each).
loaded = set(s2_all_series.iloc[:N]) | set(s3_all_series.iloc[:N]) if N else (
    s2_all | s3_all
)
del s2_all_series, s3_all_series

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
if not {"source1_entity_id", "candidate_entity_ids"} <= set(cand.columns):
    raise RuntimeError(f"Unexpected candidate columns: {cand.columns.tolist()}")

if len(cand) != len(s1_ids):
    raise RuntimeError(
        f"Candidate file has {len(cand):,} rows, expected {len(s1_ids):,}. "
        "Make sure it came from the same smoke run."
    )

if not cand["source1_entity_id"].reset_index(drop=True).equals(
    s1_ids.reset_index(drop=True)
):
    raise RuntimeError("Candidate rows are not in S1 file order.")

if gt["source1_entity_id"].duplicated().any():
    raise RuntimeError("Duplicate source1_entity_id in ground truth.")

if s2_all & s3_all:
    raise RuntimeError("S2/S3 entity_id collision; source attribution ambiguous.")

# Align GT to the candidate rows by ID, never by position.
gt_map = gt.set_index("source1_entity_id")["matched_entity_ids"]
missing = ~cand["source1_entity_id"].isin(gt_map.index)
if missing.any():
    raise RuntimeError(
        f"{int(missing.sum()):,} candidate S1 IDs are missing from GT, e.g. "
        f"{cand.loc[missing, 'source1_entity_id'].head(5).tolist()}"
    )
true_col = gt_map.reindex(cand["source1_entity_id"]).to_numpy()
del gt, gt_map

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
stats = {
    k: 0 for k in (
        "true_empty", "fp_on_true_empty", "matched", "matched_zero_cand",
        "true_pairs", "tp", "s2_true", "s2_tp", "s3_true", "s3_tp",
        "other_true", "zero_cand_rows", "n_pred", "dup_pred_rows",
        "unknown_pred_ids",
        "r_true_pairs", "r_tp", "r_s2_true", "r_s2_tp", "r_s3_true",
        "r_s3_tp", "r_matched", "r_matched_zero_cand",
    )
}
f_all = 0.0
f_reach = 0.0

for pred_raw, true_raw in zip(cand["candidate_entity_ids"].to_numpy(), true_col):
    pred_list = [v.strip() for v in pred_raw.split(",") if v.strip()]
    pred = set(pred_list)
    true = parse_ids(true_raw)
    reach = true & loaded

    if len(pred_list) != len(pred):
        stats["dup_pred_rows"] += 1
    stats["unknown_pred_ids"] += sum(
        1 for x in pred if x not in s2_all and x not in s3_all
    )
    stats["n_pred"] += len(pred)
    if not pred:
        stats["zero_cand_rows"] += 1

    hit = true & pred
    f_all += f05(len(hit), len(pred), len(true))
    f_reach += f05(len(reach & pred), len(pred), len(reach))

    if not true:
        stats["true_empty"] += 1
        if pred:
            stats["fp_on_true_empty"] += 1
        continue

    stats["matched"] += 1
    if not pred:
        stats["matched_zero_cand"] += 1
    stats["true_pairs"] += len(true)
    stats["tp"] += len(hit)
    for x in true:
        src = "s2" if x in s2_all else "s3" if x in s3_all else None
        if src is None:
            stats["other_true"] += 1
            continue
        stats[f"{src}_true"] += 1
        stats[f"{src}_tp"] += x in pred
        if x in loaded:
            stats[f"r_{src}_true"] += 1
            stats[f"r_{src}_tp"] += x in pred

    if reach:
        stats["r_matched"] += 1
        if not pred:
            stats["r_matched_zero_cand"] += 1
        stats["r_true_pairs"] += len(reach)
        stats["r_tp"] += len(reach & pred)

n = len(cand)


def ratio(a, b):
    return f"{a / b:.4%}" if b else "n/a"


print("\n===== SMOKE EVALUATION (GT aligned by source1_entity_id) =====")
print(f"S1 rows evaluated:                {n:,}")
print(f"Target IDs loaded by smoke run:   {len(loaded):,}")
print(f"Candidate pairs:                  {stats['n_pred']:,}")
print(f"Average candidates/S1:            {stats['n_pred'] / n:.4f}")
print(f"Zero-candidate S1 (all rows):     {stats['zero_cand_rows']:,}")
print(f"Rows with duplicate candidates:   {stats['dup_pred_rows']:,}")
print(f"Candidate IDs not in S2/S3:       {stats['unknown_pred_ids']:,}")
print()
print("--- ALL GT (true targets anywhere in full S2/S3) ---")
print(f"True-empty S1 rows:               {stats['true_empty']:,}")
print(f"False positives on true-empty:    {stats['fp_on_true_empty']:,}")
print(f"Matched S1 rows:                  {stats['matched']:,}")
print(f"Matched S1 with zero candidates:  {stats['matched_zero_cand']:,}")
print(f"True pairs:                       {stats['true_pairs']:,}")
print(f"True pairs retrieved (TP):        {stats['tp']:,}")
print(f"Overall pair recall:              {ratio(stats['tp'], stats['true_pairs'])}")
print(f"S2 pair recall:                   {ratio(stats['s2_tp'], stats['s2_true'])}"
      f"  ({stats['s2_tp']:,}/{stats['s2_true']:,})")
print(f"S3 pair recall:                   {ratio(stats['s3_tp'], stats['s3_true'])}"
      f"  ({stats['s3_tp']:,}/{stats['s3_true']:,})")
print(f"True IDs not in S2/S3:            {stats['other_true']:,}")
print(f"Macro F0.5 of candidate set:      {f_all / n:.6f}")
print()
print("--- REACHABLE (true targets restricted to smoke-loaded S2/S3) ---")
print(f"S1 rows with >=1 reachable match: {stats['r_matched']:,}")
print(f"  ...with zero candidates:        {stats['r_matched_zero_cand']:,}")
print(f"Reachable true pairs:             {stats['r_true_pairs']:,}")
print(f"Reachable TP:                     {stats['r_tp']:,}")
print(f"Overall pair recall:              {ratio(stats['r_tp'], stats['r_true_pairs'])}")
print(f"S2 pair recall:                   {ratio(stats['r_s2_tp'], stats['r_s2_true'])}"
      f"  ({stats['r_s2_tp']:,}/{stats['r_s2_true']:,})")
print(f"S3 pair recall:                   {ratio(stats['r_s3_tp'], stats['r_s3_true'])}"
      f"  ({stats['r_s3_tp']:,}/{stats['r_s3_true']:,})")
print(f"Macro F0.5 of candidate set:      {f_reach / n:.6f}")
print("=" * 70)
print("Smoke numbers are NOT leaderboard performance.")
