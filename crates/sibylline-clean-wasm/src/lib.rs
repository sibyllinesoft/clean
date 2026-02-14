use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Fused text normalization: NFKC + strip invisible + collapse whitespace + lowercase.
#[wasm_bindgen]
pub fn normalize_text(text: &str) -> String {
    sibylline_clean::normalize_text(text)
}

/// Compiled pattern extractor using RegexSet for O(n) multi-pattern matching.
#[wasm_bindgen]
pub struct PatternExtractor {
    inner: sibylline_clean::PatternExtractor,
}

#[wasm_bindgen]
impl PatternExtractor {
    /// Create a new extractor from a JS object of {category: [pattern_strings]}.
    #[wasm_bindgen(constructor)]
    pub fn new(patterns: JsValue) -> Result<PatternExtractor, JsError> {
        let patterns: HashMap<String, Vec<String>> = serde_wasm_bindgen::from_value(patterns)
            .map_err(|e| JsError::new(&format!("Invalid patterns object: {}", e)))?;
        let inner = sibylline_clean::PatternExtractor::new(patterns)
            .map_err(|e| JsError::new(&format!("Failed to compile RegexSet: {}", e)))?;
        Ok(Self { inner })
    }

    /// Extract features from text. Returns a JS object with category densities + text stats.
    pub fn extract(&self, text: &str) -> Result<JsValue, JsError> {
        let features = self.inner.extract(text);
        serde_wasm_bindgen::to_value(&features)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }

    /// Find all pattern match spans, merged and sorted.
    /// Returns an array of [start, end] pairs.
    pub fn find_pattern_spans(&self, text: &str) -> Result<JsValue, JsError> {
        let spans = self.inner.find_pattern_spans(text);
        serde_wasm_bindgen::to_value(&spans)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }

    /// Quick check if any pattern matches at all.
    pub fn has_any_match(&self, text: &str) -> bool {
        self.inner.has_any_match(text)
    }

    /// Find all pattern match spans with category info, NOT merged.
    /// Returns array of [start, end, category] tuples.
    pub fn find_categorized_spans(&self, text: &str) -> Result<JsValue, JsError> {
        let spans = self.inner.find_categorized_spans(text);
        serde_wasm_bindgen::to_value(&spans)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }
}

/// Motif matcher with Aho-Corasick exact matching + optional fuzzy fallback.
#[wasm_bindgen]
pub struct MotifMatcher {
    inner: sibylline_clean::MotifMatcher,
}

#[wasm_bindgen]
impl MotifMatcher {
    /// Create a new motif matcher from a JS object of {category: [motif_strings]}.
    #[wasm_bindgen(constructor)]
    pub fn new(motifs: JsValue, threshold: u32, use_fuzzy: bool) -> Result<MotifMatcher, JsError> {
        let motifs: HashMap<String, Vec<String>> = serde_wasm_bindgen::from_value(motifs)
            .map_err(|e| JsError::new(&format!("Invalid motifs object: {}", e)))?;
        let inner = sibylline_clean::MotifMatcher::new(motifs, threshold, use_fuzzy)
            .map_err(|e| JsError::new(&format!("Failed to build automaton: {}", e)))?;
        Ok(Self { inner })
    }

    /// Find all motif matches in text. Returns serialized array of MotifMatch objects.
    pub fn find_matches(
        &self,
        text: &str,
        window_size: usize,
        step: usize,
    ) -> Result<JsValue, JsError> {
        let matches = self.inner.find_matches(text, window_size, step);
        serde_wasm_bindgen::to_value(&matches)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }

    /// Get merged match position spans. Returns array of [start, end] pairs.
    pub fn get_match_positions(&self, text: &str) -> Result<JsValue, JsError> {
        let spans = self.inner.get_match_positions(text);
        serde_wasm_bindgen::to_value(&spans)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }
}

/// Extract CRF features from text. Returns serialized array of feature maps.
#[wasm_bindgen]
pub fn text_to_features(text: &str) -> Result<JsValue, JsError> {
    let features = sibylline_clean::text_to_features(text);
    serde_wasm_bindgen::to_value(&features)
        .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
}
