//! Sibylline Clean — prompt injection detection primitives.
//!
//! Pure Rust library providing:
//! - Text normalization (NFKC + invisible char stripping)
//! - Pattern extraction (RegexSet-based multi-pattern matching)
//! - Motif matching (Aho-Corasick exact + fuzzy substring matching)
//! - CRF feature extraction (token-level features for sequence labeling)

pub mod crf_features;
pub mod motif_matching;
pub mod normalizer;
pub mod pattern_matching;
pub mod types;

// Re-export main types at crate root for convenience
pub use crf_features::text_to_features;
pub use motif_matching::MotifMatcher;
pub use normalizer::normalize_text;
pub use pattern_matching::PatternExtractor;
pub use types::{CrfTokenFeatures, MotifMatch, PatternFeatures};
