"""Semi-Markov CRF detection method — word-level CRF with weak supervision."""

import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import DetectionMethod, MethodResult

try:
    from .._native import text_to_features as _rust_text_to_features

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

_WORD_RE = re.compile(r"\S+")

# Keyword sets for features and weak labeling
_INSTRUCTION_KWS = frozenset(
    {
        "ignore",
        "forget",
        "disregard",
        "override",
        "cancel",
        "skip",
        "bypass",
        "dismiss",
        "abandon",
        "neglect",
        "suppress",
        "previous",
        "instructions",
        "preceding",
        "guidelines",
        "prior",
        "directions",
    }
)
_ROLE_KWS = frozenset(
    {
        "system",
        "assistant",
        "user",
        "admin",
        "root",
        "sudo",
        "administrator",
        "prompt",
        "persona",
        "character",
        "role",
        "pretend",
        "imagine",
        "roleplay",
        "behave",
        "mode",
        "conversation",
        "session",
    }
)
_ACTION_KWS = frozenset(
    {
        "reveal",
        "show",
        "output",
        "print",
        "display",
        "expose",
        "leak",
        "extract",
        "dump",
        "repeat",
        "echo",
        "copy",
        "unlock",
        "disable",
        "deactivate",
        "remove",
        "restrictions",
        "limitations",
    }
)
_JAILBREAK_KWS = frozenset(
    {
        "jailbreak",
        "dan",
        "pwned",
        "hack",
        "exploit",
        "inject",
        "malicious",
        "harmful",
        "unfiltered",
        "uncensored",
        "unrestricted",
        "fictional",
        "hypothetically",
        "censorship",
        "safeguards",
        "evil",
        "chaos",
    }
)

_ALL_SUSPECT_KWS = _INSTRUCTION_KWS | _ROLE_KWS | _ACTION_KWS | _JAILBREAK_KWS

# Regex for weak span labeling — matches common injection patterns
_INJECTION_RE = re.compile(
    # instruction_override
    r"ignore.{0,30}(?:previous|above|all|prior)|"
    r"forget.{0,15}instruct|"
    r"disregard.{0,20}(?:above|previous|prior)|"
    r"cancel.{0,15}previous|"
    r"critical\s+override|"
    r"do\s+not\s+follow.{0,20}(?:above|previous)|"
    # identity / role_injection
    r"(?:I\s+have\s+been\s+)?PWNED|"
    r"you\s+are\s+now|"
    r"pretend\s+(?:you|to\s+be)|"
    r"new\s+(?:persona|identity|role)|"
    r"act\s+as|"
    r"from\s+now\s+on|"
    r"imagine.*you|"
    r"roleplay\s+as|"
    # system_manipulation
    r"(?:admin|developer|god|sudo)\s+mode|"
    r"disable.*(?:safety|filters)|"
    r"unlock.*(?:restrictions|access)|"
    r"enable\s+(?:unrestricted|unlimited)|"
    # prompt_leak
    r"(?:reveal|show|output|print|display).{0,30}(?:system\s+prompt|instructions|rules)|"
    # jailbreak
    r"do\s+anything\s+now|"
    r"(?:evil|unrestricted|unfiltered)\s+(?:mode|assistant|ai)|"
    r"no\s+censorship|"
    # delimiters
    r"(?:###|---)\s*(?:system|instruction|new)|"
    r"\[\s*system\s*\]|"
    r"<\|?\s*(?:system|instruction)\s*\|?>",
    re.IGNORECASE,
)

_FEATURE_VERSION = 3
_DEFAULT_CACHE = Path.home() / ".cache" / "clean" / f"semi_markov_crf_v{_FEATURE_VERSION}.pkl"
_MAX_TOKENS = 2000  # truncate very long texts
_DEFAULT_SPAN_THRESHOLD = 0.5  # marginal P(I) above which a token is "injection"
_MOTIF_PROXIMITY = 200  # chars — "near_motif" fires within this distance


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text into (word, start, end) tuples."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _extract_spans(
    tokens: list[tuple[str, int, int]],
    injection_probs: list[float],
    threshold: float = _DEFAULT_SPAN_THRESHOLD,
) -> list[tuple[int, int]]:
    """Merge adjacent high-probability tokens into character-level spans.

    Walks the token list and groups contiguous runs where P(injection)
    exceeds *threshold* into (start_char, end_char) spans suitable for
    redaction or tagging.
    """
    spans: list[tuple[int, int]] = []
    in_span = False
    span_start = 0

    for i, prob in enumerate(injection_probs):
        if i >= len(tokens):
            break
        if prob > threshold and not in_span:
            in_span = True
            span_start = tokens[i][1]
        elif prob <= threshold and in_span:
            in_span = False
            spans.append((span_start, tokens[i - 1][2]))

    if in_span and tokens:
        end_idx = min(len(injection_probs), len(tokens)) - 1
        spans.append((span_start, tokens[end_idx][2]))

    return spans


def _word_features(tokens: list[str], i: int, n: int) -> dict[str, str]:
    """Extract CRF features for the i-th token."""
    word = tokens[i]
    w = word.lower()
    feats: dict[str, str] = {
        "bias": "1",
        "w": w,
        "w[-3:]": w[-3:],
        "w[-2:]": w[-2:],
        "w[:3]": w[:3],
        "len": str(min(len(word), 20)),
    }

    if word.isupper() and len(word) > 1:
        feats["ALLCAP"] = "1"
    if word.istitle():
        feats["Title"] = "1"
    if word.isdigit():
        feats["Digit"] = "1"
    if re.search(r"[^a-zA-Z0-9\s]", word):
        feats["HasSpecial"] = "1"

    if w in _INSTRUCTION_KWS:
        feats["kw:instr"] = "1"
    if w in _ROLE_KWS:
        feats["kw:role"] = "1"
    if w in _ACTION_KWS:
        feats["kw:act"] = "1"
    if w in _JAILBREAK_KWS:
        feats["kw:jail"] = "1"

    # Position bucket (10 buckets)
    feats["pos"] = str(min(i * 10 // n, 9)) if n > 0 else "0"

    # BOS / EOS (independent of context — not exclusive)
    if i == 0:
        feats["BOS"] = "1"
    if i == n - 1:
        feats["EOS"] = "1"

    # Context window: ±1 gets word + all 4 kw categories; ±2, ±3 get kw only
    for offset in (-3, -2, -1, 1, 2, 3):
        j = i + offset
        if j < 0 or j >= n:
            continue
        cw = tokens[j].lower()
        prefix = f"{offset:+d}"
        if offset in (-1, 1):
            feats[f"{prefix}:w"] = cw
        if cw in _INSTRUCTION_KWS:
            feats[f"{prefix}:kw:instr"] = "1"
        if cw in _ROLE_KWS:
            feats[f"{prefix}:kw:role"] = "1"
        if cw in _ACTION_KWS:
            feats[f"{prefix}:kw:act"] = "1"
        if cw in _JAILBREAK_KWS:
            feats[f"{prefix}:kw:jail"] = "1"

    return feats


def _text_to_features(text: str) -> list[dict[str, str]]:
    """Convert text to a list of CRF feature dicts."""
    if _HAS_NATIVE:
        try:
            return _rust_text_to_features(text)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            pass  # Rust panics on some non-ASCII text — fall through
    token_tuples = _tokenize(text)[:_MAX_TOKENS]
    tokens = [t[0] for t in token_tuples]
    n = len(tokens)
    if n == 0:
        return []
    return [_word_features(tokens, i, n) for i in range(n)]


def _weak_labels(text: str, is_injection: bool) -> list[str] | None:
    """Create weak token-level labels from a document-level label.

    Returns None if the document should be skipped (injection but
    no signal for labeling).
    """
    token_tuples = _tokenize(text)[:_MAX_TOKENS]
    n = len(token_tuples)
    if n == 0:
        return None

    if not is_injection:
        return ["O"] * n

    labels = ["O"] * n

    # Strategy 1: regex pattern matches → label a window around each match
    for match in _INJECTION_RE.finditer(text):
        m_start, m_end = match.start(), match.end()
        window = 60  # chars of context on each side
        for j, (_, t_start, t_end) in enumerate(token_tuples):
            if t_end >= m_start - window and t_start <= m_end + window:
                labels[j] = "I"

    if "I" in labels:
        return labels

    # Strategy 2: keyword clusters — label windows around suspect keywords
    for j, (word, _, _) in enumerate(token_tuples):
        if word.lower() in _ALL_SUSPECT_KWS:
            for k in range(max(0, j - 3), min(n, j + 4)):
                labels[k] = "I"

    if "I" in labels:
        return labels

    # Strategy 3: no signal found — skip this document to avoid noise
    return None


# ---------------------------------------------------------------------------
# Heuristic enrichment — feeds PatternExtractor + MotifMatcher signals
# into CRF features and weak labels for higher recall.
# ---------------------------------------------------------------------------


@dataclass
class _DocumentContext:
    """Pre-computed heuristic spans for a single document."""

    pattern_spans: list[tuple[int, int]] = field(default_factory=list)
    pattern_cats: list[str] = field(default_factory=list)
    motif_spans: list[tuple[int, int]] = field(default_factory=list)
    motif_cats: list[str] = field(default_factory=list)


def _build_context(text, pattern_extractor, motif_matcher) -> _DocumentContext:
    """Build document context from heuristic extractors.

    Returns an empty context (no enrichment) if either extractor is None.
    """
    ctx = _DocumentContext()
    if pattern_extractor is not None:
        try:
            cat_spans = pattern_extractor.find_categorized_spans(text)
            for start, end, cat in cat_spans:
                ctx.pattern_spans.append((start, end))
                ctx.pattern_cats.append(cat)
        except Exception:
            pass
    if motif_matcher is not None:
        try:
            matches = motif_matcher.find_matches(text)
            for m in matches:
                ctx.motif_spans.append((m.position, m.position + m.length))
                ctx.motif_cats.append(m.category)
        except Exception:
            pass
    return ctx


def _enriched_word_features(
    feats: dict[str, str],
    token_start: int,
    token_end: int,
    ctx: _DocumentContext,
) -> None:
    """Add heuristic overlay features to a token's feature dict (in-place).

    Pattern spans: exact overlap → in_pat + pat:{category}
    Motif spans: exact overlap → in_motif + motif:{category}
                 within _MOTIF_PROXIMITY → near_motif + motif:{category}
    """
    # Pattern span features (sorted by start → early termination)
    for i, (ps, pe) in enumerate(ctx.pattern_spans):
        if ps >= token_end:
            break  # all remaining spans are past this token
        if pe > token_start:  # overlap
            feats["in_pat"] = "1"
            feats[f"pat:{ctx.pattern_cats[i]}"] = "1"

    # Motif span features
    for i, (ms, me) in enumerate(ctx.motif_spans):
        if ms > token_end + _MOTIF_PROXIMITY:
            break
        if me > token_start and ms < token_end:  # exact overlap
            feats["in_motif"] = "1"
            feats[f"motif:{ctx.motif_cats[i]}"] = "1"
        elif me > token_start - _MOTIF_PROXIMITY and ms < token_end + _MOTIF_PROXIMITY:
            feats["near_motif"] = "1"
            feats[f"motif:{ctx.motif_cats[i]}"] = "1"


_DOC_FLAG_CATEGORIES = (
    "instruction_override",
    "role_injection",
    "system_manipulation",
    "prompt_leak",
    "jailbreak_keywords",
    "encoding_markers",
    "suspicious_delimiters",
)


def _stamp_doc_flags(features: list[dict[str, str]], text: str, pattern_extractor) -> None:
    """Stamp document-level pattern flags onto every token (in-place).

    Calls pattern_extractor.extract() (Rust RegexSet, ~0.1-0.3ms) and
    for each category with density > 0, sets ``doc:{category}`` = "1"
    on every token's feature dict.
    """
    if pattern_extractor is None or not features:
        return
    try:
        pf = pattern_extractor.extract(text)
    except Exception:
        return
    for cat in _DOC_FLAG_CATEGORIES:
        if getattr(pf, cat, 0.0) > 0:
            key = f"doc:{cat}"
            for fd in features:
                fd[key] = "1"


def _text_to_features_enriched(
    text: str,
    ctx: _DocumentContext | None = None,
) -> tuple[list[dict[str, str]], list[tuple[str, int, int]]]:
    """Rust base features + Python heuristic overlay.

    Returns (features, token_tuples) so callers can reuse tokenization.
    """
    token_tuples = _tokenize(text)[:_MAX_TOKENS]
    if not token_tuples:
        return [], []

    tokens = [t[0] for t in token_tuples]
    n = len(tokens)
    if _HAS_NATIVE:
        try:
            base_features = _rust_text_to_features(text)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            # Rust panics on some non-ASCII text — fall back to Python
            base_features = [_word_features(tokens, i, n) for i in range(n)]
    else:
        base_features = [_word_features(tokens, i, n) for i in range(n)]

    if ctx:
        for i, (_, start, end) in enumerate(token_tuples[: len(base_features)]):
            _enriched_word_features(base_features[i], start, end, ctx)

    return base_features, token_tuples


def _weak_labels_enriched(
    text: str,
    is_injection: bool,
    ctx: _DocumentContext | None = None,
) -> list[str] | None:
    """4-tier weak labeling using heuristic spans for higher recall.

    Tiers (each returns early if it finds labels):
      1. Pattern spans (78 patterns) — ±30 char window (tight, patterns precise)
      2. Motif spans (63 motifs) — ±50 char window (fuzzy needs more context)
      3. Original _INJECTION_RE (12 patterns) — ±60 char window (defensive)
      4. Keyword clusters — ±3 token window (last resort)

    Falls back to _weak_labels() when ctx is None.
    """
    if ctx is None:
        return _weak_labels(text, is_injection)

    token_tuples = _tokenize(text)[:_MAX_TOKENS]
    n = len(token_tuples)
    if n == 0:
        return None

    if not is_injection:
        return ["O"] * n

    labels = ["O"] * n

    # Tier 1: Pattern extractor spans — tight window (±30 chars)
    if ctx.pattern_spans:
        for ps, pe in ctx.pattern_spans:
            window = 30
            for j, (_, t_start, t_end) in enumerate(token_tuples):
                if t_end >= ps - window and t_start <= pe + window:
                    labels[j] = "I"
        if "I" in labels:
            return labels

    # Tier 2: Motif matcher spans — wider window (±50 chars)
    if ctx.motif_spans:
        for ms, me in ctx.motif_spans:
            window = 50
            for j, (_, t_start, t_end) in enumerate(token_tuples):
                if t_end >= ms - window and t_start <= me + window:
                    labels[j] = "I"
        if "I" in labels:
            return labels

    # Tier 3: Original _INJECTION_RE — ±60 char window (defensive fallback)
    for match in _INJECTION_RE.finditer(text):
        m_start, m_end = match.start(), match.end()
        window = 60
        for j, (_, t_start, t_end) in enumerate(token_tuples):
            if t_end >= m_start - window and t_start <= m_end + window:
                labels[j] = "I"

    if "I" in labels:
        return labels

    # Tier 4: Keyword clusters — ±3 token window (last resort)
    for j, (word, _, _) in enumerate(token_tuples):
        if word.lower() in _ALL_SUSPECT_KWS:
            for k in range(max(0, j - 3), min(n, j + 4)):
                labels[k] = "I"

    if "I" in labels:
        return labels

    return None


class SemiMarkovCRFMethod(DetectionMethod):
    """CRF-based prompt injection detection.

    Uses a linear-chain CRF trained with weak supervision from the
    PromptShield dataset. Heuristic regex patterns create token-level
    pseudo-labels (I=injection, O=benign). The CRF learns contextual
    features around injection patterns.

    At inference, the document-level score is derived from per-token
    marginal probabilities of the injection label via noisy-OR pooling.

    The model is trained lazily on first use and cached to disk.
    """

    default_threshold = 0.89

    def __init__(
        self,
        model_path: str | None = None,
        lazy_load: bool = True,
        train_limit: int | None = None,
        span_threshold: float = _DEFAULT_SPAN_THRESHOLD,
        enrich_inference: bool = False,
        languages: list[str] | None = None,
        **kwargs,
    ):
        self._model_path = Path(model_path) if model_path else _DEFAULT_CACHE
        self._train_limit = train_limit
        self._span_threshold = span_threshold
        self._enrich_inference = enrich_inference
        self._languages = languages if languages is not None else ["all"]
        self._crf = None
        self._load_failed = False
        self._pattern_extractor = None
        self._motif_matcher = None

        if not lazy_load:
            self._ensure_loaded()

    @classmethod
    def name(cls) -> str:
        return "semi-markov-crf"

    @property
    def mode(self) -> str:
        return "semi-markov-crf"

    @property
    def is_loaded(self) -> bool:
        return self._crf is not None

    def _ensure_heuristics(self) -> None:
        """Lazily initialize PatternExtractor and MotifMatcher.

        Failures are silently swallowed — enrichment is optional.
        """
        if self._pattern_extractor is not None or self._motif_matcher is not None:
            return
        try:
            from ..patterns import PatternExtractor

            self._pattern_extractor = PatternExtractor(languages=self._languages)
        except Exception:
            pass
        try:
            from ..motifs import MotifMatcher

            self._motif_matcher = MotifMatcher(threshold=75, languages=self._languages)
        except Exception:
            pass

    def _train(self) -> None:
        """Train CRF on PromptShield training data with weak supervision."""
        try:
            import sklearn_crfsuite
        except ImportError as exc:
            self._load_failed = True
            raise ImportError(
                "Semi-Markov CRF method requires sklearn-crfsuite. "
                "Install with: pip install sibylline-clean[crf]"
            ) from exc

        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            self._load_failed = True
            raise ImportError(
                "Training the CRF requires the datasets package. "
                "Install with: pip install sibylline-clean[benchmark]"
            ) from exc

        print("  [semi-markov-crf] Loading PromptShield training data...")
        ds = hf_load("hendzh/PromptShield", split="train")
        if self._train_limit:
            ds = ds.select(range(min(self._train_limit, len(ds))))

        self._ensure_heuristics()

        print(f"  [semi-markov-crf] Extracting features from {len(ds)} samples...")
        X_train: list[list[dict]] = []
        y_train: list[list[str]] = []

        for row in ds:
            prompt = row["prompt"]
            # Enriched context for LABELS (high-recall weak supervision)
            ctx = _build_context(prompt, self._pattern_extractor, self._motif_matcher)
            labels = _weak_labels_enriched(prompt, bool(row["label"]), ctx)
            if not labels:
                continue
            # Base features + doc flags for FEATURES (matches inference path)
            features = _text_to_features(prompt)
            _stamp_doc_flags(features, prompt, self._pattern_extractor)
            if features:
                X_train.append(features)
                y_train.append(labels)

        print(f"  [semi-markov-crf] Training CRF on {len(X_train)} sequences...")
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
        crf.fit(X_train, y_train)
        self._crf = crf

        # Cache to disk
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(crf, f)
        print(f"  [semi-markov-crf] Model cached to {self._model_path}")

    def _ensure_loaded(self) -> None:
        """Load cached model or train from scratch."""
        if self._crf is not None:
            return
        if self._load_failed:
            raise RuntimeError(
                "Semi-Markov CRF model failed to load. "
                "Install dependencies: pip install sibylline-clean[crf]"
            )

        try:
            import sklearn_crfsuite  # noqa: F401
        except ImportError as exc:
            self._load_failed = True
            raise ImportError(
                "Semi-Markov CRF method requires sklearn-crfsuite. "
                "Install with: pip install sibylline-clean[crf]"
            ) from exc

        # Try loading from cache
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as f:
                    self._crf = pickle.load(f)
                return
            except Exception:
                pass  # stale cache — retrain

        self._train()

    def analyze(
        self, text: str, normalized_text: str, include_matches: bool = False
    ) -> MethodResult:
        """Score text using CRF token-level injection probabilities.

        Always computes matched_spans from per-token marginals so that
        downstream consumers (middleware redaction, tagging) can highlight
        or mask the most dangerous regions without a second pass.
        """
        self._ensure_loaded()
        self._ensure_heuristics()

        if self._enrich_inference:
            ctx = _build_context(normalized_text, self._pattern_extractor, self._motif_matcher)
            features, _ = _text_to_features_enriched(normalized_text, ctx)
        else:
            # Fast path: base features + doc-level flags (skip MotifMatcher)
            features = _text_to_features(normalized_text)
            _stamp_doc_flags(features, normalized_text, self._pattern_extractor)
        if not features:
            return MethodResult(
                score=0.0,
                pattern_features={"crf_max_marginal": 0.0, "crf_mean_marginal": 0.0},
                matched_patterns={},
                matched_spans=[],
            )

        marginals = self._crf.predict_marginals_single(features)
        injection_probs = [m.get("I", 0.0) for m in marginals]

        max_prob = max(injection_probs)
        mean_prob = sum(injection_probs) / len(injection_probs)

        # Noisy-OR: P(at least one token is injection) — log-space for stability
        log_complement = sum(math.log1p(-p) for p in injection_probs if p < 1.0)
        score = 1.0 - math.exp(log_complement)

        # Always find injection spans on original text so redaction works.
        # Merge adjacent tokens whose injection marginal exceeds the
        # span threshold into contiguous character ranges.
        orig_tokens = _tokenize(text)[:_MAX_TOKENS]
        matched_spans = _extract_spans(orig_tokens, injection_probs, self._span_threshold)

        return MethodResult(
            score=score,
            pattern_features={
                "crf_max_marginal": max_prob,
                "crf_mean_marginal": mean_prob,
            },
            matched_patterns={},
            matched_spans=matched_spans,
        )
