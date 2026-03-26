"""
Haned et al. (2011) reproduction on PROVEDIt.

Implements a likelihood-based estimator for Number Of Contributors (NOC)
using qualitative allele presence information only, then computes:
- Conditional probabilities Pr(x_hat=i | x=k)
- Predictive values Pr(x=i | x_hat=i) via Bayes theorem

The script is designed to run directly on the PROVEDIt folder structure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_RAW_DIR, MARKERS_TO_REMOVE, RESULTS_DIR


EXPERT_PRIORS = {
    "expert1_traces": [0.45, 0.04, 0.30, 0.15, 0.06],
    "expert1_body_fluids": [0.87, 0.07, 0.04, 0.01, 0.01],
    "expert2_traces": [0.45, 0.04, 0.35, 0.15, 0.01],
    "expert2_body_fluids": [0.87, 0.07, 0.05, 0.01, 0.00],
    "expert3_traces": [0.45, 0.04, 0.25, 0.20, 0.06],
    "expert3_body_fluids": [0.87, 0.07, 0.05, 0.01, 0.00],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Haned 2011 on PROVEDIt")
    parser.add_argument("--data-root", type=str, default=DATA_RAW_DIR, help="Path to PROVEDIt root folder")
    parser.add_argument("--output-dir", type=str, default=os.path.join(RESULTS_DIR, "haned2011"))
    parser.add_argument("--max-contributors", type=int, default=5, help="Max true contributors (K)")
    parser.add_argument("--search-max", type=int, default=6, help="Max candidate contributors for estimator")
    parser.add_argument("--simulations", type=int, default=5000, help="Monte Carlo simulations per marker per n")
    parser.add_argument("--max-alleles", type=int, default=30, help="Use Allele 1..max_alleles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument(
        "--multiplex",
        type=str,
        default="all",
        help="Comma-separated keywords to filter folders/files, e.g. IDPlus28,GF29; or 'all'",
    )
    parser.add_argument(
        "--injection-times",
        type=str,
        default="all",
        help="Comma-separated filter, e.g. '10 sec,15 sec'; or 'all'",
    )
    return parser.parse_args()


def _canonical_allele(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "OL":
        return None
    try:
        numeric = float(text)
        return f"{numeric:g}"
    except ValueError:
        return text


def _extract_noc_from_sample_name(sample_name: str) -> int:
    match = re.search(r"RD\d+-\d+-([0-9]+(?:_[0-9]+)*)-", sample_name)
    if match:
        return len(match.group(1).split("_"))
    match_alt = re.search(r"-M(\d)", sample_name)
    if match_alt:
        return int(match_alt.group(1))
    return -1


def _extract_noc_from_path(path: Path) -> int | None:
    text = str(path)
    if "/1-Person/" in text:
        return 1
    explicit = re.search(r"/(\d)-Person/", text)
    if explicit:
        return int(explicit.group(1))
    return None


def _iter_csv_files(data_root: str) -> list[Path]:
    return sorted(Path(data_root).rglob("*.csv"))


def _matches_filters(path: Path, multiplex_filters: list[str], injection_filters: list[str]) -> bool:
    text = str(path)
    if multiplex_filters and not any(token in text for token in multiplex_filters):
        return False
    if injection_filters and not any(token in text for token in injection_filters):
        return False
    return True


def _row_unique_allele_count(row: pd.Series, allele_cols: list[str]) -> int:
    alleles = set()
    for col in allele_cols:
        allele = _canonical_allele(row[col])
        if allele is not None:
            alleles.add(allele)
    return len(alleles)


def load_provedit_qualitative_data(
    data_root: str,
    max_alleles: int,
    max_contributors: int,
    multiplex_filters: list[str],
    injection_filters: list[str],
    max_files: int,
) -> tuple[pd.DataFrame, dict[str, Counter], dict[str, int]]:
    csv_files = _iter_csv_files(data_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_root}")

    filtered_files = [p for p in csv_files if _matches_filters(p, multiplex_filters, injection_filters)]
    if max_files > 0:
        filtered_files = filtered_files[:max_files]
    if not filtered_files:
        raise ValueError("No CSV files matched filters.")

    allele_counters: dict[str, Counter] = defaultdict(Counter)
    rows = []
    stats = {"files": 0, "rows": 0, "profiles": 0}

    for idx, csv_path in enumerate(filtered_files, start=1):
        usecols = ["Sample File", "Marker"] + [f"Allele {i}" for i in range(1, max_alleles + 1)]
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in usecols, low_memory=False)
        except ValueError:
            continue

        if df.empty:
            continue

        df = df[~df["Marker"].isin(MARKERS_TO_REMOVE)].copy()
        if df.empty:
            continue

        fixed_noc = _extract_noc_from_path(csv_path)
        if fixed_noc is not None:
            df["NOC"] = fixed_noc
        else:
            df["NOC"] = df["Sample File"].astype(str).map(_extract_noc_from_sample_name)
        df = df[df["NOC"].between(1, max_contributors)]
        if df.empty:
            continue

        allele_cols = [c for c in df.columns if c.startswith("Allele ")]
        if not allele_cols:
            continue

        df["allele_count"] = df.apply(lambda r: _row_unique_allele_count(r, allele_cols), axis=1)
        df = df[df["allele_count"] > 0]
        if df.empty:
            continue

        df["profile_id"] = csv_path.stem + "::" + df["Sample File"].astype(str)

        single_source = df[df["NOC"] == 1]
        if not single_source.empty:
            for col in allele_cols:
                canonical = single_source[col].map(_canonical_allele)
                valid = canonical.notna()
                if not valid.any():
                    continue
                grouped = pd.DataFrame(
                    {
                        "Marker": single_source.loc[valid, "Marker"].astype(str).values,
                        "allele": canonical.loc[valid].values,
                    }
                )
                for marker, sub in grouped.groupby("Marker"):
                    allele_counters[str(marker)].update(sub["allele"].tolist())

        rows.append(df[["profile_id", "NOC", "Marker", "allele_count"]])

        stats["files"] += 1
        stats["rows"] += len(df)
        stats["profiles"] += df["profile_id"].nunique()

        if idx % 20 == 0:
            print(f"Loaded {idx}/{len(filtered_files)} files...")

    if not rows:
        raise ValueError("No valid PROVEDIt records were loaded.")

    merged = pd.concat(rows, ignore_index=True)
    stats["profiles"] = merged["profile_id"].nunique()
    return merged, allele_counters, stats


def simulate_marker_probabilities(
    allele_counters: dict[str, Counter],
    search_max: int,
    simulations: int,
    seed: int,
) -> dict[str, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    probability_tables: dict[str, dict[int, np.ndarray]] = {}

    for marker, counter in allele_counters.items():
        total = sum(counter.values())
        if total == 0:
            continue

        allele_ids = np.arange(len(counter), dtype=int)
        probs = np.array(list(counter.values()), dtype=float) / total

        per_n = {}
        for n in range(1, search_max + 1):
            draws = rng.choice(allele_ids, size=(simulations, 2 * n), p=probs)
            sorted_draws = np.sort(draws, axis=1)
            unique_counts = 1 + np.sum(np.diff(sorted_draws, axis=1) != 0, axis=1)
            pmf = np.bincount(unique_counts, minlength=(2 * n + 1)).astype(float)
            pmf /= pmf.sum()
            per_n[n] = pmf

        probability_tables[marker] = per_n

    if not probability_tables:
        raise ValueError("No marker probability tables created. Check single-source data coverage.")

    return probability_tables


def build_profile_matrix(df: pd.DataFrame, markers: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    filtered = df[df["Marker"].isin(markers)].copy()
    profile_labels = filtered.groupby("profile_id")["NOC"].first()
    matrix = filtered.pivot_table(
        index="profile_id",
        columns="Marker",
        values="allele_count",
        aggfunc="max",
        fill_value=0,
    )
    matrix = matrix.reindex(columns=markers, fill_value=0)
    matrix = matrix.loc[profile_labels.index]

    x = matrix.to_numpy(dtype=int)
    y = profile_labels.to_numpy(dtype=int)
    profile_ids = list(profile_labels.index)
    return x, y, profile_ids


def estimate_noc_mle(
    x: np.ndarray,
    markers: list[str],
    probability_tables: dict[str, dict[int, np.ndarray]],
    search_max: int,
    epsilon: float = 1e-9,
) -> np.ndarray:
    n_profiles, n_markers = x.shape
    log_likelihoods = np.zeros((n_profiles, search_max), dtype=float)

    for n in range(1, search_max + 1):
        total = np.zeros(n_profiles, dtype=float)
        for marker_idx in range(n_markers):
            marker = markers[marker_idx]
            pmf = probability_tables[marker][n]
            counts = x[:, marker_idx]
            p = np.full(n_profiles, epsilon, dtype=float)
            valid = (counts >= 1) & (counts < len(pmf))
            p[valid] = pmf[counts[valid]]
            total += np.log(np.clip(p, epsilon, None))
        log_likelihoods[:, n - 1] = total

    return np.argmax(log_likelihoods, axis=1) + 1


def estimate_noc_max_allele_count(x: np.ndarray, search_max: int) -> np.ndarray:
    max_per_profile = x.max(axis=1)
    estimates = np.ceil(max_per_profile / 2.0).astype(int)
    return np.clip(estimates, 1, search_max)


def conditional_matrix(y_true: np.ndarray, y_pred: np.ndarray, max_true: int, max_pred: int) -> np.ndarray:
    matrix = np.zeros((max_true, max_pred), dtype=float)
    for true_label in range(1, max_true + 1):
        mask = y_true == true_label
        if mask.sum() == 0:
            continue
        counts = np.bincount(y_pred[mask], minlength=max_pred + 1)[1 : max_pred + 1]
        matrix[true_label - 1] = counts / counts.sum()
    return matrix


def compute_predictive_values(cond_probs: np.ndarray, prior: list[float], max_contributors: int) -> list[float]:
    prior_arr = np.array(prior, dtype=float)
    prior_arr = prior_arr / prior_arr.sum()

    values = []
    for i in range(1, max_contributors + 1):
        numerator = cond_probs[i - 1, i - 1] * prior_arr[i - 1]
        denominator = np.sum(cond_probs[:max_contributors, i - 1] * prior_arr)
        values.append(float(numerator / denominator) if denominator > 0 else float("nan"))
    return values


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    multiplex_filters = [] if args.multiplex.lower() == "all" else [s.strip() for s in args.multiplex.split(",") if s.strip()]
    injection_filters = [] if args.injection_times.lower() == "all" else [s.strip() for s in args.injection_times.split(",") if s.strip()]

    print("Loading PROVEDIt qualitative data...")
    df, allele_counters, stats = load_provedit_qualitative_data(
        data_root=args.data_root,
        max_alleles=args.max_alleles,
        max_contributors=args.max_contributors,
        multiplex_filters=multiplex_filters,
        injection_filters=injection_filters,
        max_files=args.max_files,
    )

    print(f"Loaded: {stats['files']} files, {stats['profiles']} profiles, {stats['rows']} marker-rows")
    print(f"Markers with single-source allele frequencies: {len(allele_counters)}")

    print("Simulating Pr(observed_alleles | n contributors, marker)...")
    probability_tables = simulate_marker_probabilities(
        allele_counters=allele_counters,
        search_max=args.search_max,
        simulations=args.simulations,
        seed=args.seed,
    )

    markers = sorted(probability_tables.keys())
    x, y, profile_ids = build_profile_matrix(df, markers)

    print(f"Evaluating estimator on {len(y)} PROVEDIt profiles...")
    y_pred_mle = estimate_noc_mle(
        x=x,
        markers=markers,
        probability_tables=probability_tables,
        search_max=args.search_max,
    )
    y_pred_mac = estimate_noc_max_allele_count(x, args.search_max)

    mle_acc = accuracy(y, y_pred_mle)
    mac_acc = accuracy(y, y_pred_mac)

    cond_mle = conditional_matrix(y, y_pred_mle, args.max_contributors, args.search_max)
    cond_mac = conditional_matrix(y, y_pred_mac, args.max_contributors, args.search_max)

    priors = dict(EXPERT_PRIORS)
    empirical_prior = [(y == i).mean() for i in range(1, args.max_contributors + 1)]
    priors["provedit_empirical"] = empirical_prior

    pv_rows = []
    for name, prior in priors.items():
        pv_mle = compute_predictive_values(cond_mle, prior, args.max_contributors)
        pv_mac = compute_predictive_values(cond_mac, prior, args.max_contributors)
        for noc in range(1, args.max_contributors + 1):
            pv_rows.append(
                {
                    "prior": name,
                    "noc": noc,
                    "pv_mle": pv_mle[noc - 1],
                    "pv_mac": pv_mac[noc - 1],
                }
            )

    columns_pred = [f"xhat_{i}" for i in range(1, args.search_max + 1)]
    cond_mle_df = pd.DataFrame(cond_mle, columns=columns_pred)
    cond_mle_df.insert(0, "x_true", np.arange(1, args.max_contributors + 1))
    cond_mac_df = pd.DataFrame(cond_mac, columns=columns_pred)
    cond_mac_df.insert(0, "x_true", np.arange(1, args.max_contributors + 1))
    pv_df = pd.DataFrame(pv_rows)

    predictions_df = pd.DataFrame(
        {
            "profile_id": profile_ids,
            "x_true": y,
            "xhat_mle": y_pred_mle,
            "xhat_mac": y_pred_mac,
        }
    )

    cond_mle_path = os.path.join(args.output_dir, "conditional_probs_mle.csv")
    cond_mac_path = os.path.join(args.output_dir, "conditional_probs_mac.csv")
    pv_path = os.path.join(args.output_dir, "predictive_values.csv")
    pred_path = os.path.join(args.output_dir, "profile_predictions.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    cond_mle_df.to_csv(cond_mle_path, index=False)
    cond_mac_df.to_csv(cond_mac_path, index=False)
    pv_df.to_csv(pv_path, index=False)
    predictions_df.to_csv(pred_path, index=False)

    summary = {
        "accuracy_mle": mle_acc,
        "accuracy_max_allele_count": mac_acc,
        "n_profiles": int(len(y)),
        "n_markers": int(len(markers)),
        "n_files": int(stats["files"]),
        "simulations_per_marker": int(args.simulations),
        "max_true_contributors": int(args.max_contributors),
        "search_max": int(args.search_max),
        "data_root": args.data_root,
        "multiplex_filter": args.multiplex,
        "injection_filter": args.injection_times,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Haned 2011 Reproduction Completed ===")
    print(f"MLE accuracy: {mle_acc:.4f}")
    print(f"MAC accuracy: {mac_acc:.4f}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved conditional matrix (MLE): {cond_mle_path}")
    print(f"Saved conditional matrix (MAC): {cond_mac_path}")
    print(f"Saved predictive values: {pv_path}")
    print(f"Saved profile predictions: {pred_path}")


if __name__ == "__main__":
    main()
