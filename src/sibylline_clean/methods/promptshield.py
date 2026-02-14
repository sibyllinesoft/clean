"""PromptShield detection method — transformer-based classifier."""

from .base import DetectionMethod, MethodResult

# Default: ProtectAI's DeBERTa-v3 prompt injection detector (ungated).
# For Meta's gated Llama Prompt Guard, pass
# model_name="meta-llama/Llama-Prompt-Guard-2-86M" (requires HF auth).
_DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


class PromptShieldMethod(DetectionMethod):
    """Transformer-based prompt injection detection.

    Uses a HuggingFace sequence classification model (default:
    protectai/deberta-v3-base-prompt-injection-v2) to score text for
    prompt injection. The model is lazy-loaded on the first analyze()
    call.

    The model_name is configurable for experimentation — e.g. swap in
    ``meta-llama/Llama-Prompt-Guard-2-86M`` or any binary classifier
    whose label index 1 is the injection/malicious class.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        lazy_load: bool = True,
        device: str | None = None,
        **kwargs,
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._load_failed = False

        if not lazy_load:
            self._ensure_loaded()

    @classmethod
    def name(cls) -> str:
        return "promptshield"

    @property
    def mode(self) -> str:
        return "transformer"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> str:
        """Pick CUDA when available, otherwise CPU."""
        if self._device is not None:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError(
                "PromptShield model failed to load previously. "
                "Install dependencies: pip install sibylline-clean[promptshield]"
            )

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            self._load_failed = True
            raise ImportError(
                "PromptShield method requires torch and transformers. "
                "Install them with: pip install sibylline-clean[promptshield]"
            ) from exc

        device = self._resolve_device()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
            ).to(device)
        except OSError as exc:
            self._load_failed = True
            raise OSError(
                f"Failed to download model {self._model_name!r}. "
                "If the repo is gated, set HF_TOKEN or run `huggingface-cli login`."
            ) from exc
        self._model.eval()
        self._device = device

    def analyze(
        self, text: str, normalized_text: str, include_matches: bool = False
    ) -> MethodResult:
        """Score text with the transformer classifier.

        Uses the raw text (not normalized) since the transformer has its
        own tokenizer.  Returns a MethodResult with score equal to the
        softmax probability of the MALICIOUS class (index 1).
        """
        import torch
        from torch.nn import functional as F

        self._ensure_loaded()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = F.softmax(logits, dim=-1)
        malicious_score = probs[0, 1].item()

        return MethodResult(
            score=malicious_score,
            pattern_features={"transformer": malicious_score},
            matched_patterns={},
            matched_spans=[],
        )
