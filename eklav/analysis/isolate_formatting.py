import json
import os
from collections import defaultdict

BASE = os.environ.get("EKLAV_OOD_EVAL_RESULTS_DIR", "./ood_eval_results_v2")
BENCHMARKS = [
    "FeTaQARetrieval", "NQTablesRetrieval", "OTTQASmallRetrieval", "OpenWikiTablesRetrieval",
]
METHODS = ["NaiveCot", "CotGen", "CotCond"]  # naive, std-sft, eklav


def load(benchmark, method):
    path = os.path.join(BASE, benchmark, method, "full_eval.jsonl")
    rows = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows[d["qid"]] = d
    return rows


def main():
    grand_rows = []  # per-benchmark summary rows
    common_ndcg_rows = []

    for benchmark in BENCHMARKS:
        data = {m: load(benchmark, m) for m in METHODS}
        n_total = len(data[METHODS[0]])

        # sanity: same qid set across methods for this benchmark
        qid_sets = {m: set(data[m].keys()) for m in METHODS}
        common_qids_all = set.intersection(*qid_sets.values())

        # per-method: number/fraction with correct (parseable) format
        correct_counts = {}
        for m in METHODS:
            n_correct = sum(1 for d in data[m].values() if not d["parse_failed"])
            correct_counts[m] = n_correct

        # common set: qids where ALL THREE methods parsed correctly
        common_correct_qids = set(qid_sets[METHODS[0]])
        for m in METHODS:
            correct_qids_m = {qid for qid, d in data[m].items() if not d["parse_failed"]}
            common_correct_qids &= correct_qids_m

        grand_rows.append({
            "benchmark": benchmark,
            "n_total": n_total,
            **{f"{m}_n_correct": correct_counts[m] for m in METHODS},
            **{f"{m}_pct_correct": round(100 * correct_counts[m] / n_total, 1) for m in METHODS},
            "n_common_correct": len(common_correct_qids),
            "pct_common_correct": round(100 * len(common_correct_qids) / n_total, 1),
        })

        # recompute pooled nDCG@10 restricted to common_correct_qids, for each method,
        # using the already-saved per-question metrics (no re-generation needed)
        ndcg_row = {"benchmark": benchmark, "n_common": len(common_correct_qids)}
        for m in METHODS:
            if common_correct_qids:
                vals = [data[m][qid]["metrics"]["ndcg@10"] for qid in common_correct_qids]
                ndcg_row[f"{m}_ndcg10_common"] = round(sum(vals) / len(vals), 4)
            else:
                ndcg_row[f"{m}_ndcg10_common"] = None
            # also the original full-set pooled nDCG@10, for reference/delta
            vals_full = [d["metrics"]["ndcg@10"] for d in data[m].values()]
            ndcg_row[f"{m}_ndcg10_full"] = round(sum(vals_full) / len(vals_full), 4)
        common_ndcg_rows.append(ndcg_row)

    # ---- print formatting-compliance table ----
    print("=" * 110)
    print("TABLE 1: Format-compliance counts per benchmark (n questions with parseable/valid output)")
    print("=" * 110)
    hdr = f"{'Benchmark':<24}{'N':>6}" + "".join(f"{m+' n':>12}{m+' %':>8}" for m in METHODS) + f"{'Common':>10}{'Common%':>10}"
    print(hdr)
    for r in grand_rows:
        line = f"{r['benchmark']:<24}{r['n_total']:>6}"
        for m in METHODS:
            line += f"{r[f'{m}_n_correct']:>12}{r[f'{m}_pct_correct']:>8}"
        line += f"{r['n_common_correct']:>10}{r['pct_common_correct']:>10}"
        print(line)

    # totals
    tot_total = sum(r["n_total"] for r in grand_rows)
    tot_common = sum(r["n_common_correct"] for r in grand_rows)
    print("-" * 110)
    tot_line = f"{'TOTAL (pooled)':<24}{tot_total:>6}"
    for m in METHODS:
        tot_correct = sum(r[f"{m}_n_correct"] for r in grand_rows)
        tot_line += f"{tot_correct:>12}{round(100*tot_correct/tot_total,1):>8}"
    tot_line += f"{tot_common:>10}{round(100*tot_common/tot_total,1):>10}"
    print(tot_line)

    print()
    print("=" * 110)
    print("TABLE 2: nDCG@10 on the COMMON subset (qids where NaiveCot, CotGen, AND CotCond all parsed correctly)")
    print("         vs. original full-set nDCG@10 (all questions, using each row's saved per-question metric)")
    print("=" * 110)
    hdr2 = f"{'Benchmark':<24}{'N common':>10}"
    for m in METHODS:
        hdr2 += f"{m+' full':>12}{m+' common':>14}"
    print(hdr2)
    for r in common_ndcg_rows:
        line = f"{r['benchmark']:<24}{r['n_common']:>10}"
        for m in METHODS:
            line += f"{r[f'{m}_ndcg10_full']:>12}{r[f'{m}_ndcg10_common']:>14}"
        print(line)

    # pooled (macro-avg across benchmarks) common-subset nDCG
    print("-" * 110)
    pooled_line = f"{'MACRO-AVG across benchmarks':<24}{'':>10}"
    for m in METHODS:
        full_vals = [r[f"{m}_ndcg10_full"] for r in common_ndcg_rows]
        common_vals = [r[f"{m}_ndcg10_common"] for r in common_ndcg_rows if r[f"{m}_ndcg10_common"] is not None]
        pooled_line += f"{round(sum(full_vals)/len(full_vals),4):>12}{round(sum(common_vals)/len(common_vals),4):>14}"
    print(pooled_line)

    # deltas: CotCond - CotGen, full vs common
    print()
    print("=" * 110)
    print("TABLE 3: Eklav (CotCond) advantage over std-SFT (CotGen), full-set vs. common-subset nDCG@10")
    print("=" * 110)
    hdr3 = f"{'Benchmark':<24}{'Delta (full)':>14}{'Delta (common)':>16}{'Shrinkage':>12}"
    print(hdr3)
    full_deltas, common_deltas = [], []
    for r in common_ndcg_rows:
        d_full = r["CotCond_ndcg10_full"] - r["CotGen_ndcg10_full"]
        if r["CotCond_ndcg10_common"] is not None:
            d_common = r["CotCond_ndcg10_common"] - r["CotGen_ndcg10_common"]
            shrink = round(100 * (1 - d_common / d_full), 1) if d_full != 0 else float("nan")
        else:
            d_common = None
            shrink = None
        full_deltas.append(d_full)
        if d_common is not None:
            common_deltas.append(d_common)
        print(f"{r['benchmark']:<24}{round(d_full,4):>14}{'' if d_common is None else round(d_common,4):>16}{'' if shrink is None else shrink:>12}")
    print("-" * 110)
    avg_full = sum(full_deltas) / len(full_deltas)
    avg_common = sum(common_deltas) / len(common_deltas) if common_deltas else None
    print(f"{'MACRO-AVG':<24}{round(avg_full,4):>14}{'' if avg_common is None else round(avg_common,4):>16}")


if __name__ == "__main__":
    main()
