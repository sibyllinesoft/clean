#![allow(clippy::useless_conversion)] // PyO3 generates conversions via proc macros

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Fused text normalization: NFKC + strip invisible + collapse whitespace + lowercase.
#[pyfunction]
fn normalize_text(text: &str) -> String {
    sibylline_clean::normalize_text(text)
}

/// Compiled pattern extractor using RegexSet for O(n) multi-pattern matching.
#[pyclass]
struct RustPatternExtractor {
    inner: sibylline_clean::PatternExtractor,
}

#[pymethods]
impl RustPatternExtractor {
    /// Create a new extractor from a dict of {category: [pattern_strings]}.
    #[new]
    fn new(patterns: HashMap<String, Vec<String>>) -> PyResult<Self> {
        let inner = sibylline_clean::PatternExtractor::new(patterns).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to compile RegexSet: {}",
                e
            ))
        })?;
        Ok(Self { inner })
    }

    /// Extract features as a dict matching PatternFeatures field names.
    fn extract<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let features = self.inner.extract(text);
        let dict = PyDict::new_bound(py);

        // Pattern densities
        for (cat, density) in &features.category_densities {
            dict.set_item(cat.as_str(), *density)?;
        }

        // Text statistics
        dict.set_item("text_length", features.text_length)?;
        dict.set_item("special_char_ratio", features.special_char_ratio)?;
        dict.set_item("caps_ratio", features.caps_ratio)?;
        dict.set_item("newline_density", features.newline_density)?;
        dict.set_item("avg_word_length", features.avg_word_length)?;

        Ok(dict)
    }

    /// Find all pattern match spans, merged and sorted.
    fn find_pattern_spans(&self, text: &str) -> Vec<(usize, usize)> {
        self.inner.find_pattern_spans(text)
    }

    /// Quick check if any pattern matches at all.
    fn has_any_match(&self, text: &str) -> bool {
        self.inner.has_any_match(text)
    }

    /// Find all pattern match spans with category info, NOT merged.
    fn find_categorized_spans(&self, text: &str) -> Vec<(usize, usize, String)> {
        self.inner.find_categorized_spans(text)
    }
}

/// Motif matcher with Aho-Corasick exact matching + optional fuzzy fallback.
#[pyclass]
struct RustMotifMatcher {
    inner: sibylline_clean::MotifMatcher,
}

#[pymethods]
impl RustMotifMatcher {
    /// Create a new motif matcher.
    #[new]
    fn new(
        motifs: HashMap<String, Vec<String>>,
        threshold: u32,
        use_fuzzy: bool,
    ) -> PyResult<Self> {
        let inner =
            sibylline_clean::MotifMatcher::new(motifs, threshold, use_fuzzy).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to build Aho-Corasick automaton: {}",
                    e
                ))
            })?;
        Ok(Self { inner })
    }

    /// Find all motif matches in text.
    fn find_matches<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        window_size: usize,
        step: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let matches = self.inner.find_matches(text, window_size, step);
        let mut results: Vec<Bound<'py, PyDict>> = Vec::with_capacity(matches.len());

        for m in matches {
            let dict = PyDict::new_bound(py);
            dict.set_item("motif", m.motif.as_str())?;
            dict.set_item("category", m.category.as_str())?;
            dict.set_item("position", m.position)?;
            dict.set_item("length", m.length)?;
            dict.set_item("score", m.score)?;
            results.push(dict);
        }

        Ok(results)
    }

    /// Get merged match position spans.
    fn get_match_positions(&self, text: &str) -> Vec<(usize, usize)> {
        self.inner.get_match_positions(text)
    }
}

/// Extract CRF features from text, returning a list of feature dicts.
#[pyfunction]
fn text_to_features<'py>(py: Python<'py>, text: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let features = sibylline_clean::text_to_features(text);
    let mut results: Vec<Bound<'py, PyDict>> = Vec::with_capacity(features.len());

    for feature_map in features {
        let dict = PyDict::new_bound(py);
        for (key, value) in &feature_map {
            dict.set_item(key.as_str(), value.as_str())?;
        }
        results.push(dict);
    }

    Ok(results)
}

/// Native accelerator module for sibylline-clean.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    m.add_class::<RustPatternExtractor>()?;
    m.add_class::<RustMotifMatcher>()?;
    m.add_function(wrap_pyfunction!(text_to_features, m)?)?;
    Ok(())
}
