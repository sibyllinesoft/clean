# Benchmarks

Evaluation infrastructure for comparing Clean's detection methods against each other and against published results from other prompt injection detectors.

## Dataset

Benchmarks run against [hendzh/PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) (ACM CODASPY 2025), a curated dataset with 23,516 test samples spanning conversational and application-structured injection patterns. The dataset is significant because it uses structurally novel injection patterns -- different link phrases and attack strategies between train and test splits -- which exposes classifiers that memorize training data rather than learning attack structure.

## Running benchmarks

```bash
# Install benchmark dependencies
pip install 'sibylline-clean[benchmark]'

# Run all registered methods
python -m sibylline_clean.benchmarks.promptshield

# Run a specific method
python -m sibylline_clean.benchmarks.promptshield --method heuristic
python -m sibylline_clean.benchmarks.promptshield --method semi-markov-crf
python -m sibylline_clean.benchmarks.promptshield --method promptshield

# Quick test with limited samples
python -m sibylline_clean.benchmarks.promptshield --limit 500

# Pattern-only mode (skip embedding classifier)
python -m sibylline_clean.benchmarks.promptshield --no-embeddings

# Custom threshold and output directory
python -m sibylline_clean.benchmarks.promptshield --threshold 0.5 --output-dir my_results/

# Print results without saving
python -m sibylline_clean.benchmarks.promptshield --no-save
```

## Metrics

The benchmark computes the metrics that matter for production deployment:

- **AUC** -- Area under the ROC curve. Threshold-independent measure of overall discriminative ability.
- **TPR @ FPR** -- True positive rate (recall) at specific false positive rate thresholds (1%, 0.5%, 0.1%, 0.05%). This is the critical metric: it answers "how many attacks do I catch if I can only tolerate N% of benign inputs being falsely flagged?" Low-FPR evaluation is essential because false positives are expensive in production.
- **Precision / Recall / F1** -- Standard classification metrics at the method's decision threshold.

## Results directory

Benchmark results are saved as timestamped JSON files in `results/`:

```
results/
  heuristic_20260215_140350.json
  semi_markov_crf_20260215_140246.json
  promptshield_20260215_140601.json
```

Each file contains the full metrics, runtime, configuration, and error information. Results are tagged with the Clean version for reproducibility.

## Interpreting results

The key numbers to compare are **TPR @ 1% FPR** and **TPR @ 0.5% FPR**. These tell you how well a detector performs under realistic production constraints where false positives have real costs (user friction, blocked legitimate requests, alert fatigue).

High F1 or accuracy numbers can be misleading -- a model can achieve strong F1 at a threshold that produces unacceptable false positive rates. Always look at TPR at specific FPR operating points.

## Code structure

| File | Purpose |
|------|---------|
| `src/sibylline_clean/benchmarks/promptshield.py` | Dataset loading, method runner, result formatting, CLI |
| `src/sibylline_clean/benchmarks/metrics.py` | AUC, TPR@FPR, precision/recall/F1 computation |
| `src/sibylline_clean/benchmarks/__main__.py` | Entry point for `python -m` invocation |

## Adding a new benchmark dataset

1. Add a new runner module in `src/sibylline_clean/benchmarks/` following the `promptshield.py` pattern
2. Implement `load_dataset()` to return `(prompts, labels, langs)` lists
3. Use `metrics.compute_metrics()` for standardized evaluation
4. Use `save_result()` to write results to the output directory
