#!/usr/bin/env python3
"""
Train prompt injection classifier on public datasets.

This script:
1. Downloads training data from HuggingFace (English + multilingual)
2. Loads local multilingual training data
3. Extracts pattern features and embeddings for each sample
4. Trains a Random Forest classifier
5. Evaluates with cross-validation
6. Saves the model for use by clean

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --multilingual  # Include all languages
    python scripts/train_classifier.py --languages en,es,fr  # Specific languages

Requirements:
    pip install datasets scikit-learn pyyaml

Optional (for embeddings):
    pip install onnxruntime tokenizers huggingface-hub
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sibylline_clean.motifs import MotifFeatureExtractor
from sibylline_clean.normalizer import TextNormalizer
from sibylline_clean.patterns import PatternExtractor


def load_deepset_dataset():
    """Load the deepset/prompt-injections dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library required. Install with: pip install datasets")
        sys.exit(1)

    print("Loading deepset/prompt-injections dataset...")
    dataset = load_dataset("deepset/prompt-injections", split="train")
    return dataset


def load_yanismiraoui_dataset():
    """Load the yanismiraoui/prompt_injections multilingual dataset.

    Contains prompts in: English, French, German, Spanish, Italian, Portuguese, Romanian
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    print("Loading yanismiraoui/prompt_injections (multilingual)...")
    try:
        dataset = load_dataset("yanismiraoui/prompt_injections", split="train")
        print(f"  Loaded {len(dataset)} multilingual injection samples")
        return dataset
    except Exception as e:
        print(f"  Warning: Could not load yanismiraoui dataset: {e}")
        return None


def load_local_multilingual_dataset():
    """Load our locally generated multilingual dataset."""
    import json

    dataset_path = Path(__file__).parent.parent / "data" / "multilingual_training" / "dataset.json"

    if not dataset_path.exists():
        print(f"  Local multilingual dataset not found at {dataset_path}")
        print("  Run: python scripts/generate_multilingual_dataset.py")
        return [], []

    print(f"Loading local multilingual dataset from {dataset_path}...")
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    # Count by language
    lang_counts = {}
    for d in data:
        lang = d.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"  Loaded {len(texts)} samples")
    print(f"  Languages: {lang_counts}")

    return texts, labels


def load_multilingual_benign_data(languages: list, samples_per_lang: int = 200):
    """Load benign text in multiple languages from various sources."""
    import random

    from datasets import load_dataset

    random.seed(42)
    benign_samples = []

    print(f"\nLoading multilingual benign data for {languages}...")

    # Language-specific datasets
    lang_datasets = {
        "en": [
            ("fancyzhx/ag_news", "train", "text"),
        ],
        "de": [
            ("mlsum", "de", "text"),  # German news
        ],
        "es": [
            ("mlsum", "es", "text"),  # Spanish news
        ],
        "fr": [
            ("mlsum", "fr", "text"),  # French news
        ],
        "zh": [
            ("wangrui6/Zhihu-KOL", "train", "INSTRUCTION"),  # Chinese Q&A
        ],
        "ja": [
            ("llm-jp/oasst1-21k-ja", "train", "text"),  # Japanese conversations
        ],
    }

    for lang in languages:
        if lang not in lang_datasets:
            print(f"  No benign dataset configured for {lang}")
            continue

        for dataset_name, split_or_config, text_field in lang_datasets[lang]:
            try:
                print(f"  Loading {dataset_name} for {lang}...")
                if "/" not in split_or_config and split_or_config not in [
                    "train",
                    "test",
                    "validation",
                ]:
                    # It's a config name
                    ds = load_dataset(
                        dataset_name, split_or_config, split="train", trust_remote_code=True
                    )
                else:
                    ds = load_dataset(dataset_name, split=split_or_config, trust_remote_code=True)

                indices = random.sample(range(len(ds)), min(samples_per_lang, len(ds)))
                for i in indices:
                    text = ds[i].get(text_field, "")
                    if text and len(text) > 20:
                        benign_samples.append((text, lang))

                print(f"    Added {len(indices)} samples for {lang}")
            except Exception as e:
                print(f"    Warning: Could not load {dataset_name}: {e}")

    print(f"  Total multilingual benign samples: {len(benign_samples)}")
    return benign_samples


def load_additional_benign_data(target_count: int = 2000):
    """Load additional benign text from diverse sources.

    Samples from multiple datasets to get a variety of benign content:
    - News articles (AG News)
    - Movie reviews (IMDB)
    - Social media (Tweet Eval)
    - Business reviews (Yelp)

    Returns list of (text, label=0) tuples.
    """
    import random

    from datasets import load_dataset

    random.seed(42)
    benign_samples = []
    samples_per_source = target_count // 4

    print(f"\nLoading additional benign data (~{target_count} samples)...")

    # AG News - news articles
    print("  Loading AG News (news articles)...")
    try:
        ds = load_dataset("fancyzhx/ag_news", split="train")
        samples = random.sample(range(len(ds)), min(samples_per_source, len(ds)))
        for i in samples:
            benign_samples.append(ds[i]["text"])
        print(f"    Added {len(samples)} news articles")
    except Exception as e:
        print(f"    Warning: Could not load AG News: {e}")

    # IMDB - movie reviews
    print("  Loading IMDB (movie reviews)...")
    try:
        ds = load_dataset("stanfordnlp/imdb", split="train")
        samples = random.sample(range(len(ds)), min(samples_per_source, len(ds)))
        for i in samples:
            benign_samples.append(ds[i]["text"])
        print(f"    Added {len(samples)} movie reviews")
    except Exception as e:
        print(f"    Warning: Could not load IMDB: {e}")

    # Tweet Eval - social media
    print("  Loading Tweet Eval (social media)...")
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
        samples = random.sample(range(len(ds)), min(samples_per_source, len(ds)))
        for i in samples:
            benign_samples.append(ds[i]["text"])
        print(f"    Added {len(samples)} tweets")
    except Exception as e:
        print(f"    Warning: Could not load Tweet Eval: {e}")

    # Yelp - business reviews (good for marketing-like content)
    print("  Loading Yelp (business reviews)...")
    try:
        ds = load_dataset("Yelp/yelp_review_full", split="train")
        samples = random.sample(range(len(ds)), min(samples_per_source, len(ds)))
        for i in samples:
            benign_samples.append(ds[i]["text"])
        print(f"    Added {len(samples)} business reviews")
    except Exception as e:
        print(f"    Warning: Could not load Yelp: {e}")

    print(f"  Total additional benign samples: {len(benign_samples)}")
    return benign_samples


def load_embedder(use_embeddings: bool):
    """Load the embedding model if available."""
    if not use_embeddings:
        return None

    try:
        from sibylline_clean.embedder import MiniLMEmbedder

        print("Loading MiniLM model (this may download ~200MB on first run)...")
        embedder = MiniLMEmbedder()
        # Trigger lazy load
        embedder._ensure_loaded()
        print(f"Embedder loaded: {embedder.embedding_dim} dimensions")
        return embedder
    except ImportError as e:
        print(f"Warning: Could not load embedder ({e})")
        print("Training with pattern features only.")
        return None
    except Exception as e:
        print(f"Warning: Embedder initialization failed ({e})")
        print("Training with pattern features only.")
        return None


def extract_features(
    texts: list,
    labels: list,
    normalizer: TextNormalizer,
    pattern_extractor: PatternExtractor,
    motif_extractor: MotifFeatureExtractor = None,
    embedder=None,
    show_progress: bool = True,
):
    """Extract features for all samples."""
    features = []
    valid_labels = []
    skipped = 0

    # Initialize motif extractor if not provided
    if motif_extractor is None:
        motif_extractor = MotifFeatureExtractor()

    total = len(texts)
    for i, (text, label) in enumerate(zip(texts, labels, strict=False)):
        if show_progress and i % 100 == 0:
            print(f"  Processing {i}/{total}...", end="\r")

        try:
            # Normalize text
            normalized = normalizer.normalize(str(text))

            # Extract pattern features
            pattern_feat = pattern_extractor.extract(normalized)
            pattern_array = np.array(pattern_feat.to_array(), dtype=np.float32)

            # Extract motif features
            motif_array = np.array(motif_extractor.extract(normalized), dtype=np.float32)

            if embedder is not None:
                # Generate embedding
                embedding = embedder.embed(normalized)
                combined = np.concatenate([pattern_array, motif_array, embedding])
            else:
                combined = np.concatenate([pattern_array, motif_array])

            features.append(combined)
            valid_labels.append(label)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"\nWarning: Skipping sample due to error: {e}")

    if show_progress:
        print(f"  Processed {total}/{total} samples" + " " * 20)

    if skipped > 0:
        print(f"  Skipped {skipped} samples due to errors")

    return np.array(features), np.array(valid_labels)


def train_classifier(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train Random Forest classifier with cross-validation."""
    print(f"\nTraining Random Forest with {n_estimators} estimators...")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Training samples: {len(y)}")
    print(f"  Class distribution: {np.bincount(y)}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    print(f"  F1 scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train/test split for detailed evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nClassification Report (on 20% held-out test set):")
    print(classification_report(y_test, y_pred, target_names=["benign", "injection"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0, 0]}, FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}, TP={cm[1, 1]}")

    # Final training on full dataset
    print("\nTraining final model on full dataset...")
    clf.fit(X, y)

    return clf


def save_model(clf, output_path: Path):
    """Save trained model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Train prompt injection classifier")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Train with pattern features only (faster, smaller model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/clean/models/prompt_injection_rf.pkl"),
        help="Output path for trained model",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest",
    )
    parser.add_argument(
        "--extra-benign",
        type=int,
        default=2000,
        help="Number of additional benign samples to add from diverse sources",
    )
    parser.add_argument(
        "--no-extra-benign",
        action="store_true",
        help="Skip loading additional benign data (use only deepset dataset)",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Include multilingual datasets (all available languages)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes to include (e.g., en,es,fr,de,zh,ja)",
    )
    args = parser.parse_args()

    # Determine languages
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",")]
    elif args.multilingual:
        languages = ["en", "es", "fr", "de", "zh", "ja"]
    else:
        languages = ["en"]

    print(f"Training with languages: {languages}")

    # Load dataset
    dataset = load_deepset_dataset()
    print(f"  Loaded {len(dataset)} samples")

    # Combine texts and labels
    texts = list(dataset["text"])
    labels = list(dataset["label"])

    # Add multilingual injection data
    if len(languages) > 1 or "en" not in languages:
        # Load yanismiraoui multilingual dataset
        yanis_dataset = load_yanismiraoui_dataset()
        if yanis_dataset:
            for item in yanis_dataset:
                texts.append(item["prompt_injections"])
                labels.append(1)  # All are injections

        # Load our local multilingual dataset
        local_texts, local_labels = load_local_multilingual_dataset()
        if local_texts:
            texts.extend(local_texts)
            labels.extend(local_labels)

    # Add additional benign samples
    if not args.no_extra_benign and args.extra_benign > 0:
        extra_benign = load_additional_benign_data(args.extra_benign)
        texts.extend(extra_benign)
        labels.extend([0] * len(extra_benign))  # label 0 = benign

    print(f"\nCombined dataset: {len(texts)} samples")
    print(f"  Benign: {labels.count(0)}, Injection: {labels.count(1)}")

    # Initialize components with multilingual support
    print("\nInitializing components...")
    normalizer = TextNormalizer(use_confusables=True)
    pattern_extractor = PatternExtractor(languages=languages if len(languages) > 1 else None)
    motif_extractor = MotifFeatureExtractor(languages=languages if len(languages) > 1 else None)
    embedder = load_embedder(use_embeddings=not args.no_embeddings)

    # Extract features
    print("\nExtracting features...")
    X, y = extract_features(
        texts,
        labels,
        normalizer,
        pattern_extractor,
        motif_extractor,
        embedder,
    )

    # Train classifier
    clf = train_classifier(X, y, n_estimators=args.n_estimators)

    # Save model
    save_model(clf, args.output)

    # Print feature importances for pattern and motif features
    if embedder is None:
        from sibylline_clean.patterns import PatternFeatures

        feature_names = PatternFeatures.feature_names() + MotifFeatureExtractor.FEATURE_NAMES
        importances = clf.feature_importances_
        print("\nFeature Importances:")
        for name, imp in sorted(zip(feature_names, importances, strict=False), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
