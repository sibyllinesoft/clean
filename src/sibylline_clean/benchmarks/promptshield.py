"""PromptShield benchmark runner.

Evaluates detection methods against the PromptShield dataset
(ACM CODASPY 2025, ``hendzh/PromptShield`` on HuggingFace).

Usage::

    python -m clean.benchmarks.promptshield --limit 500 --no-embeddings
    python -m clean.benchmarks.promptshield --method heuristic
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import compute_metrics

DATASET_ID = "hendzh/PromptShield"
BENCHMARK_NAME = "PromptShield"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(split: str = "test", limit: int | None = None):
    """Load the PromptShield dataset from HuggingFace.

    Returns:
        Tuple of (prompts, labels, langs) lists.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print(
            "ERROR: The 'datasets' package is required.\n"
            "Install with:  pip install 'clean-injection[benchmark]'",
            file=sys.stderr,
        )
        sys.exit(1)

    ds = hf_load_dataset(DATASET_ID, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    prompts = ds["prompt"]
    labels = ds["label"]
    langs = ds["lang"] if "lang" in ds.column_names else ["unknown"] * len(prompts)

    return prompts, labels, langs


# ---------------------------------------------------------------------------
# Single-method runner
# ---------------------------------------------------------------------------


def run_method(
    method_name: str,
    prompts: list[str],
    labels: list[int],
    threshold: float | None,
    use_embeddings: bool,
) -> dict:
    """Run a single detection method against the dataset.

    Returns a result dict with metrics, timing, and error info.
    """
    from sibylline_clean.detector import InjectionDetector

    # Try to instantiate the detector — some methods may not be implemented
    try:
        detector = InjectionDetector(
            method=method_name,
            threshold=threshold,
            use_embeddings=use_embeddings,
        )
        threshold = detector.threshold  # resolve method default
        mode = detector.mode
    except NotImplementedError as exc:
        return {
            "method": method_name,
            "mode": "(error)",
            "threshold": threshold,
            "use_embeddings": use_embeddings,
            "metrics": None,
            "runtime_seconds": 0.0,
            "num_samples": len(prompts),
            "num_errors": 0,
            "error": f"NotImplementedError: {exc}",
        }

    scores: list[float] = []
    num_errors = 0
    first_error: str | None = None

    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        try:
            result = detector.analyze(prompt)
            scores.append(result.score)
        except NotImplementedError as exc:
            # Method skeleton — abort early
            elapsed = time.perf_counter() - t0
            return {
                "method": method_name,
                "mode": mode,
                "threshold": threshold,
                "use_embeddings": use_embeddings,
                "metrics": None,
                "runtime_seconds": round(elapsed, 3),
                "num_samples": len(prompts),
                "num_errors": 0,
                "error": f"NotImplementedError: {exc}",
            }
        except Exception as exc:
            scores.append(0.0)
            num_errors += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

        if (i + 1) % 1000 == 0:
            print(f"  [{method_name}] {i + 1}/{len(prompts)} samples processed")

    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(labels, scores, threshold)

    result_dict = {
        "method": method_name,
        "mode": mode,
        "threshold": threshold,
        "use_embeddings": use_embeddings,
        "metrics": asdict(metrics),
        "runtime_seconds": round(elapsed, 3),
        "num_samples": len(prompts),
        "num_errors": num_errors,
        "error": first_error,
    }
    return result_dict


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    methods: list[str],
    split: str = "test",
    limit: int | None = None,
    threshold: float | None = None,
    use_embeddings: bool = True,
    output_dir: str | Path = "benchmarks/results",
    save: bool = True,
) -> list[dict]:
    """Run the benchmark for one or more methods.

    Returns list of result dicts.
    """
    prompts, labels, langs = load_dataset(split, limit)
    n = len(prompts)
    print(f"\n{BENCHMARK_NAME} Benchmark — {split} split ({n:,} samples)")
    print("=" * 55)

    results = []
    for method_name in methods:
        print(f"\nRunning: {method_name}")
        result = run_method(method_name, prompts, labels, threshold, use_embeddings)
        results.append(result)

    print("\n" + format_table(results))

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            path = save_result(r, out)
            print(f"Saved: {path}")
        print()

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_table(results: list[dict]) -> str:
    """Render results as an ASCII table."""
    header = (
        f"{'Method':<14} {'Mode':<14} {'AUC':>5}  {'TPR@1%':>6}  {'TPR@0.5%':>8}  "
        f"{'TPR@0.1%':>8}  {'TPR@0.05%':>9}  {'P':>5}  {'R':>5}  {'F1':>5}  {'Time':>6}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        m = r.get("metrics")
        if m is None:
            # Error row
            name = r["method"][:14]
            error_msg = r.get("error", "unknown error")
            lines.append(
                f"{name:<14} {'(error)':<14} {'—':>5}  {'—':>6}  {'—':>8}  "
                f"{'—':>8}  {'—':>9}  {'—':>5}  {'—':>5}  {'—':>5}  {'0.0s':>6}"
            )
            lines.append(f"{'':14} {error_msg}")
            continue

        tpr = m["tpr_at_fpr"]
        elapsed = f"{r['runtime_seconds']:.1f}s"
        lines.append(
            f"{r['method']:<14} {r['mode']:<14} "
            f"{m['auc']:5.3f}  {tpr.get('1%', 0):6.3f}  {tpr.get('0.5%', 0):8.3f}  "
            f"{tpr.get('0.1%', 0):8.3f}  {tpr.get('0.05%', 0):9.3f}  "
            f"{m['precision']:5.2f}  {m['recall']:5.2f}  {m['f1']:5.2f}  {elapsed:>6}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_result(result: dict, output_dir: str | Path) -> Path:
    """Write a result dict to a timestamped JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    method = result["method"].replace("-", "_")
    path = output_dir / f"{method}_{ts}.json"

    try:
        from importlib.metadata import version as pkg_version

        clean_version = pkg_version("clean-injection")
    except Exception:
        clean_version = "unknown"

    payload = {
        "benchmark": BENCHMARK_NAME,
        "dataset": DATASET_ID,
        "split": result.get("split", "test"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clean_version": clean_version,
        **result,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    from sibylline_clean.methods import list_methods

    parser = argparse.ArgumentParser(
        description="Run PromptShield prompt-injection benchmark",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Specific detection method to benchmark (default: all registered)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of samples for quick testing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold (default: method-specific or 0.3)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Pattern-only mode (skip embedding classifier)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Results directory (default: benchmarks/results)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print only, skip writing result files",
    )

    args = parser.parse_args()

    if args.method:
        methods = [args.method]
    else:
        methods = list_methods()

    run_benchmark(
        methods=methods,
        split=args.split,
        limit=args.limit,
        threshold=args.threshold,
        use_embeddings=not args.no_embeddings,
        output_dir=args.output_dir,
        save=not args.no_save,
    )
