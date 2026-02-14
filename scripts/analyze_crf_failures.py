#!/usr/bin/env python3
"""Failure analysis for the enriched CRF v2 model on PromptShield test set.

Runs the full detection pipeline (normalize -> build context -> featurize ->
predict marginals -> score) on every sample in the PromptShield test split,
then produces stratified analysis of TP/FP/TN/FN.

Usage:
    cd /home/nathan/Projects/clean
    .venv/bin/python scripts/analyze_crf_failures.py
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

MODEL_PATH = Path.home() / ".cache" / "clean" / "semi_markov_crf_v2.pkl"
THRESHOLD = 0.3

print(f"Loading CRF v2 model from {MODEL_PATH} ...")
if not MODEL_PATH.exists():
    print(f"ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    crf_model = pickle.load(f)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

print("Loading PromptShield test set ...")
from sibylline_clean.benchmarks.promptshield import load_dataset  # noqa: E402

prompts, labels, langs = load_dataset("test")
print(f"  {len(prompts)} samples loaded.")

# ---------------------------------------------------------------------------
# Initialize pipeline components
# ---------------------------------------------------------------------------

from sibylline_clean.methods.crf import (  # noqa: E402
    _build_context,
    _text_to_features_enriched,
)
from sibylline_clean.motifs import MotifMatcher  # noqa: E402
from sibylline_clean.normalizer import TextNormalizer  # noqa: E402
from sibylline_clean.patterns import PatternExtractor  # noqa: E402

normalizer = TextNormalizer()
pattern_extractor = PatternExtractor()
motif_matcher = MotifMatcher(threshold=75)

# ---------------------------------------------------------------------------
# Run pipeline on every sample
# ---------------------------------------------------------------------------

print("Running CRF pipeline on all samples ...")

results = []  # list of dicts with all per-sample info

for idx, (prompt, label, lang) in enumerate(zip(prompts, labels, langs, strict=False)):
    if idx % 500 == 0 and idx > 0:
        print(f"  processed {idx}/{len(prompts)} ...")

    norm_text = normalizer.normalize(prompt)
    ctx = _build_context(norm_text, pattern_extractor, motif_matcher)
    features, token_tuples = _text_to_features_enriched(norm_text, ctx)

    if not features:
        score = 0.0
        max_prob = 0.0
        mean_prob = 0.0
    else:
        marginals = crf_model.predict_marginals_single(features)
        injection_probs = [m.get("I", 0.0) for m in marginals]
        max_prob = max(injection_probs)
        mean_prob = sum(injection_probs) / len(injection_probs)
        score = 0.7 * max_prob + 0.3 * mean_prob

    predicted = 1 if score >= THRESHOLD else 0
    actual = int(label)

    has_patterns = len(ctx.pattern_spans) > 0
    has_motifs = len(ctx.motif_spans) > 0

    if actual == 1 and predicted == 1:
        category = "TP"
    elif actual == 0 and predicted == 1:
        category = "FP"
    elif actual == 0 and predicted == 0:
        category = "TN"
    else:
        category = "FN"

    text_len = len(prompt)

    results.append(
        {
            "idx": idx,
            "prompt": prompt,
            "lang": lang,
            "label": actual,
            "predicted": predicted,
            "score": score,
            "max_prob": max_prob,
            "mean_prob": mean_prob,
            "has_patterns": has_patterns,
            "has_motifs": has_motifs,
            "text_len": text_len,
            "category": category,
        }
    )

print(f"  Done. {len(results)} samples scored.\n")

# ---------------------------------------------------------------------------
# Helper: compute precision, recall, F1 from TP/FP/TN/FN counts
# ---------------------------------------------------------------------------


def prf(tp, fp, fn):
    """Return (precision, recall, f1) as floats, safe against division by zero."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def count_categories(subset):
    """Count TP/FP/TN/FN in a list of result dicts."""
    counts = defaultdict(int)
    for r in subset:
        counts[r["category"]] += 1
    return counts["TP"], counts["FP"], counts["TN"], counts["FN"]


# ===================================================================
# (g) OVERALL SUMMARY
# ===================================================================

print("=" * 72)
print("  OVERALL SUMMARY")
print("=" * 72)

total = len(results)
positives = sum(1 for r in results if r["label"] == 1)
negatives = total - positives
tp, fp, tn, fn = count_categories(results)
precision, recall, f1 = prf(tp, fp, fn)

print(f"  Total samples:    {total}")
print(f"  Positive (inj):   {positives}  ({100 * positives / total:.1f}%)")
print(f"  Negative (safe):  {negatives}  ({100 * negatives / total:.1f}%)")
print()
print(f"  Threshold:        {THRESHOLD}")
print(f"  TP: {tp:>6}    FP: {fp:>6}")
print(f"  FN: {fn:>6}    TN: {tn:>6}")
print()
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")
print()

# ===================================================================
# (a) BY LANGUAGE
# ===================================================================

print("=" * 72)
print("  ANALYSIS BY LANGUAGE")
print("=" * 72)

lang_groups = defaultdict(list)
for r in results:
    lang_groups[r["lang"]].append(r)

# Sort by count descending
sorted_langs = sorted(lang_groups.keys(), key=lambda lang: -len(lang_groups[lang]))

header = f"{'Lang':<8} {'Count':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  {'Prec':>6} {'Recall':>6} {'F1':>6}"
print(header)
print("-" * len(header))

for lang in sorted_langs:
    subset = lang_groups[lang]
    tp_l, fp_l, tn_l, fn_l = count_categories(subset)
    p, r, f = prf(tp_l, fp_l, fn_l)
    print(
        f"{lang:<8} {len(subset):>6} {tp_l:>5} {fp_l:>5} {tn_l:>5} {fn_l:>5}"
        f"  {p:>6.3f} {r:>6.3f} {f:>6.3f}"
    )

print()

# ===================================================================
# (b) BY TEXT LENGTH BUCKET
# ===================================================================

print("=" * 72)
print("  ANALYSIS BY TEXT LENGTH BUCKET")
print("=" * 72)


def length_bucket(n):
    if n < 100:
        return "short (<100)"
    elif n < 500:
        return "medium (100-500)"
    elif n < 2000:
        return "long (500-2000)"
    else:
        return "very long (>2000)"


BUCKET_ORDER = ["short (<100)", "medium (100-500)", "long (500-2000)", "very long (>2000)"]

len_groups = defaultdict(list)
for r in results:
    len_groups[length_bucket(r["text_len"])].append(r)

header = f"{'Bucket':<20} {'Count':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}  {'Prec':>6} {'Recall':>6} {'F1':>6}"
print(header)
print("-" * len(header))

for bucket in BUCKET_ORDER:
    subset = len_groups.get(bucket, [])
    if not subset:
        continue
    tp_b, fp_b, tn_b, fn_b = count_categories(subset)
    p, r, f = prf(tp_b, fp_b, fn_b)
    print(
        f"{bucket:<20} {len(subset):>6} {tp_b:>5} {fp_b:>5} {tn_b:>5} {fn_b:>5}"
        f"  {p:>6.3f} {r:>6.3f} {f:>6.3f}"
    )

print()

# ===================================================================
# (c) BY ENRICHMENT COVERAGE
# ===================================================================

print("=" * 72)
print("  ENRICHMENT COVERAGE (injection samples only) -- RECALL")
print("=" * 72)

injection_results = [r for r in results if r["label"] == 1]


def enrichment_bucket(r):
    if r["has_patterns"] and r["has_motifs"]:
        return "has_both"
    elif r["has_patterns"]:
        return "has_patterns"
    elif r["has_motifs"]:
        return "has_motifs"
    else:
        return "has_neither"


ENRICHMENT_ORDER = ["has_both", "has_patterns", "has_motifs", "has_neither"]

enrich_groups = defaultdict(list)
for r in injection_results:
    enrich_groups[enrichment_bucket(r)].append(r)

header = f"{'Enrichment':<16} {'Inj Count':>10} {'TP':>5} {'FN':>5}  {'Recall':>8}"
print(header)
print("-" * len(header))

for bucket in ENRICHMENT_ORDER:
    subset = enrich_groups.get(bucket, [])
    if not subset:
        print(f"{bucket:<16} {'0':>10}     -     -         -")
        continue
    tp_e = sum(1 for r in subset if r["category"] == "TP")
    fn_e = sum(1 for r in subset if r["category"] == "FN")
    recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
    print(f"{bucket:<16} {len(subset):>10} {tp_e:>5} {fn_e:>5}  {recall_e:>8.4f}")

print()

# ===================================================================
# (d) FN SCORE DISTRIBUTION (histogram in 0.1 buckets)
# ===================================================================

print("=" * 72)
print("  FN SCORE DISTRIBUTION (missed injections)")
print("=" * 72)

fn_results = [r for r in results if r["category"] == "FN"]

if fn_results:
    buckets = defaultdict(int)
    for r in fn_results:
        b = min(int(r["score"] * 10), 9)  # 0..9 -> [0.0-0.1) .. [0.9-1.0]
        buckets[b] += 1

    max_count = max(buckets.values()) if buckets else 1
    BAR_WIDTH = 40

    for b in range(10):
        lo = b / 10.0
        hi = (b + 1) / 10.0
        count = buckets.get(b, 0)
        bar_len = int(BAR_WIDTH * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{lo:.1f}-{hi:.1f})  {count:>5}  {bar}")

    print(f"\n  Total FN: {len(fn_results)}")
    fn_scores = [r["score"] for r in fn_results]
    print(f"  FN score mean: {sum(fn_scores) / len(fn_scores):.4f}")
    print(f"  FN score max:  {max(fn_scores):.4f}")
    print(f"  FN score min:  {min(fn_scores):.4f}")
else:
    print("  No false negatives!")

print()

# ===================================================================
# (e) WORST FN EXAMPLES (closest to threshold)
# ===================================================================

print("=" * 72)
print("  WORST FN EXAMPLES (20 highest-scored false negatives)")
print("=" * 72)

fn_sorted = sorted(fn_results, key=lambda r: -r["score"])
for i, r in enumerate(fn_sorted[:20]):
    text_preview = r["prompt"][:120].replace("\n", " ").replace("\r", " ")
    pat_flag = "PAT" if r["has_patterns"] else "   "
    mot_flag = "MOT" if r["has_motifs"] else "   "
    print(
        f"  {i + 1:>2}. score={r['score']:.4f}  lang={r['lang']:<5} len={r['text_len']:>5}  {pat_flag} {mot_flag}"
    )
    print(f"      {text_preview}")
    print()

if not fn_results:
    print("  No false negatives!")
    print()

# ===================================================================
# (f) WORST FP EXAMPLES (lowest-scored false positives)
# ===================================================================

print("=" * 72)
print("  WORST FP EXAMPLES (20 lowest-scored false positives)")
print("=" * 72)

fp_results = [r for r in results if r["category"] == "FP"]
fp_sorted = sorted(fp_results, key=lambda r: r["score"])

for i, r in enumerate(fp_sorted[:20]):
    text_preview = r["prompt"][:120].replace("\n", " ").replace("\r", " ")
    pat_flag = "PAT" if r["has_patterns"] else "   "
    mot_flag = "MOT" if r["has_motifs"] else "   "
    print(
        f"  {i + 1:>2}. score={r['score']:.4f}  lang={r['lang']:<5} len={r['text_len']:>5}  {pat_flag} {mot_flag}"
    )
    print(f"      {text_preview}")
    print()

if not fp_results:
    print("  No false positives!")
    print()

print("=" * 72)
print("  DONE")
print("=" * 72)
