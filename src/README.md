# Source Code Guide

Architecture overview and module map for the Clean prompt injection detection library.

## Package structure

```
src/sibylline_clean/
    __init__.py           # Public API, lazy imports for content module
    detector.py           # InjectionDetector -- main entry point
    normalizer.py         # Unicode normalization pipeline
    patterns.py           # Regex-based pattern extraction (7 categories)
    motifs.py             # Fuzzy motif matching via RapidFuzz/Aho-Corasick
    classifier.py         # Random Forest + pattern-only classifiers
    embedder.py           # MiniLM-L6-v2 ONNX embeddings
    windowing.py          # Two-phase sliding window hotspot detection
    config.py             # Multilingual YAML pattern loader

    methods/              # Pluggable detection method registry
        base.py           # DetectionMethod ABC + MethodResult dataclass
        __init__.py       # Registry (register_method / get_method / list_methods)
        heuristic.py      # Pattern + motif + optional ML pipeline
        crf.py            # Semi-Markov CRF with weak supervision
        promptshield.py   # Transformer classifier wrapper (DeBERTa / Llama)

    content/              # Content-type-aware scanning
        scanner.py        # ContentScanner orchestrator (extract -> detect -> map -> annotate)
        extractors.py     # String extractors for JSON, CSV, XML, YAML
        spans.py          # Virtual text construction + offset mapping
        heat.py           # Sub-span severity scoring
        annotators.py     # Redaction + annotation (structured / CLI / plain)

    pattern_data/         # Multilingual pattern databases
        _defaults/        # YAML files: en.yaml, es.yaml, fr.yaml, ... (13 languages)

    models/               # Bundled model weights
        prompt_injection_rf.pkl   # Pre-trained Random Forest classifier

    benchmarks/           # Evaluation infrastructure
        metrics.py        # AUC, TPR@FPR, P/R/F1 computation
        promptshield.py   # PromptShield dataset benchmark runner

    _native.pyi           # Type stubs for the Rust extension module
    _native.*.so          # Compiled Rust accelerator (built by maturin)
```

## Detection pipeline

The core detection flow through `InjectionDetector.analyze()`:

```
Input text
  -> TextNormalizer.normalize()          # NFKC + strip invisible + lowercase
  -> DetectionMethod.analyze()           # Dispatches to selected method
      -> PatternExtractor.extract()      # Regex category scores
      -> MotifMatcher.compute_signal()   # Fuzzy fragment matching
      -> [SlidingWindowAnalyzer]         # Hotspot detection for long text
      -> [MiniLMEmbedder + RF]           # Optional ML classification
  -> InjectionAnalysis                   # Score, flagged, pattern_features, matched_spans
```

## Key abstractions

### DetectionMethod (`methods/base.py`)

Abstract base class for detection strategies. Each method implements `analyze()` and is registered by name in the method registry. To add a new method:

```python
from sibylline_clean.methods import register_method
from sibylline_clean.methods.base import DetectionMethod, MethodResult

@register_method
class MyMethod(DetectionMethod):
    @classmethod
    def name(cls) -> str:
        return "my-method"

    def analyze(self, text, normalized_text, include_matches=False) -> MethodResult:
        # Your detection logic here
        return MethodResult(score=0.0, pattern_features={}, matched_patterns={}, matched_spans=[])

    @property
    def mode(self) -> str:
        return "my-mode"

    @property
    def is_loaded(self) -> bool:
        return True
```

### ContentScanner (`content/scanner.py`)

Orchestrates structured document scanning:

1. Selects an extractor by MIME content type
2. Extracts all strings with byte offsets into a `SpanMap`
3. Runs `InjectionDetector` on the concatenated virtual text
4. Scores sub-spans by category severity (`HeatScorer`)
5. Maps hot spans back to original document byte positions
6. Annotates/redacts via mode-specific formatter

### PatternConfig (`config.py`)

Loads pattern and motif definitions from YAML with priority resolution:

1. `~/.config/{app_name}/patterns/` -- user overrides (highest priority)
2. `.{app_name}/patterns/` -- project-specific patterns
3. Package defaults -- shipped with Clean (fallback)

This lets users extend the pattern database without modifying the package.

## Rust native module

The `_native` extension module (built by maturin from `crates/`) provides:

- `normalize_text()` -- fused NFKC + strip + collapse + lowercase in one pass
- `RustPatternExtractor` -- multi-pattern extraction via compiled RegexSet
- `RustMotifMatcher` -- Aho-Corasick exact substring matching
- `text_to_features()` -- CRF feature extraction

The Python code always has a pure-Python fallback path when the native module isn't available. The Rust crate also compiles to a WASM target (`crates/sibylline-clean-wasm/`) for browser deployment.

## Crate structure

```
crates/
    sibylline-clean/           # Core Rust library
        src/lib.rs             # Module declarations
        src/normalizer.rs      # Unicode normalization
        src/pattern_matching.rs # RegexSet pattern extractor
        src/motif_matching.rs  # Aho-Corasick motif matcher
        src/crf_features.rs    # CRF feature extraction
        src/types.rs           # Shared types

    sibylline-clean-python/    # PyO3 bindings (built by maturin)
        src/lib.rs             # #[pymodule] entry point

    sibylline-clean-wasm/      # wasm-bindgen target
        src/lib.rs             # #[wasm_bindgen] entry point
```

## Tests

```
tests/
    unit/                      # Fast, isolated tests for individual components
        test_normalizer.py     # Unicode normalization edge cases
        test_patterns.py       # Pattern extraction and matching
        test_motifs.py         # Motif matching and signal aggregation
        test_windowing.py      # Sliding window and hotspot detection
        test_crf_spans.py      # CRF span extraction
        test_content_scanner.py # Content scanning pipeline
        test_registry.py       # Method registry

    integration/               # End-to-end and cross-component tests
        test_detector.py       # Full detection pipeline
        test_methods.py        # All registered methods
        test_multilingual.py   # Multi-language detection
        test_content_integration.py # Structured document scanning
        test_benchmark_metrics.py   # Benchmark metric computation

    parity/                    # Rust/Python equivalence tests
        test_native_parity.py  # Ensures native module matches Python behavior
```

Run tests with: `pytest tests/`
