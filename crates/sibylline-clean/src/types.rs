use std::collections::HashMap;

/// Extracted pattern features: category densities and text statistics.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PatternFeatures {
    /// Density (matches per 1000 chars, capped at 1.0) for each pattern category.
    pub category_densities: HashMap<String, f64>,
    /// Text length / 10000, capped at 1.0.
    pub text_length: f64,
    /// Ratio of special characters to total characters.
    pub special_char_ratio: f64,
    /// Ratio of uppercase to alphabetic characters.
    pub caps_ratio: f64,
    /// Newlines / total characters.
    pub newline_density: f64,
    /// Average word length / 20, capped at 1.0.
    pub avg_word_length: f64,
}

/// A single motif match found in text.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MotifMatch {
    /// The motif string that matched (lowercased).
    pub motif: String,
    /// The category this motif belongs to.
    pub category: String,
    /// Byte offset of the match in the text.
    pub position: usize,
    /// Length of the matched region in bytes.
    pub length: usize,
    /// Match score (0-100).
    pub score: f64,
}

/// CRF token features: a map of feature name to feature value (all strings).
pub type CrfTokenFeatures = HashMap<String, String>;
