use aho_corasick::AhoCorasick;
use std::collections::{HashMap, HashSet};

use crate::types::MotifMatch;

/// Motif matcher with Aho-Corasick exact matching + optional fuzzy fallback.
pub struct MotifMatcher {
    /// Aho-Corasick automaton for exact substring matching
    ac: AhoCorasick,
    /// Flat list of (motif_lowercase, category)
    motifs: Vec<(String, String)>,
    /// Minimum fuzzy match score (0-100)
    threshold: u32,
    /// Whether to use fuzzy matching for unmatched motifs
    use_fuzzy: bool,
}

/// Compute a partial_ratio-like score: best ratio of shorter in longer.
///
/// Slides a window of `shorter.len()` over `longer` and returns the best
/// normalized LCS-based similarity score (0-100).
fn partial_ratio(shorter: &str, longer: &str) -> u32 {
    let s_chars: Vec<char> = shorter.chars().collect();
    let l_chars: Vec<char> = longer.chars().collect();

    if s_chars.is_empty() || l_chars.is_empty() {
        return 0;
    }

    if s_chars.len() > l_chars.len() {
        // Swap: always slide the shorter over the longer
        return partial_ratio(longer, shorter);
    }

    let s_len = s_chars.len();
    let l_len = l_chars.len();
    let mut best: u32 = 0;

    for start in 0..=(l_len - s_len) {
        let window = &l_chars[start..start + s_len];
        let score = ratio_chars(&s_chars, window);
        if score > best {
            best = score;
            if best == 100 {
                return 100;
            }
        }
    }

    best
}

/// Compute similarity ratio between two char slices (0-100).
/// Uses simple matching character count (not full LCS for speed).
fn ratio_chars(a: &[char], b: &[char]) -> u32 {
    if a.is_empty() && b.is_empty() {
        return 100;
    }
    let total = a.len() + b.len();
    if total == 0 {
        return 100;
    }

    // Count matching characters using a simple approach:
    // For each position, check if chars match (optimized for same-length comparison)
    let matches = if a.len() == b.len() {
        // Same length: count positional matches + shifted matches
        let direct: usize = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        // Also count chars that exist in both (bag-of-chars overlap)
        let bag = bag_overlap(a, b);
        // Use the better of the two
        direct.max(bag)
    } else {
        bag_overlap(a, b)
    };

    ((2 * matches * 100) / total) as u32
}

/// Count overlapping characters (bag/multiset intersection).
fn bag_overlap(a: &[char], b: &[char]) -> usize {
    let mut counts: HashMap<char, i32> = HashMap::new();
    for &c in a {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut overlap = 0usize;
    for &c in b {
        let entry = counts.entry(c).or_insert(0);
        if *entry > 0 {
            *entry -= 1;
            overlap += 1;
        }
    }
    overlap
}

/// Find the best starting char offset for a motif within a window string.
fn find_best_match_pos(window: &str, motif: &str) -> usize {
    // Try exact substring first
    if let Some(idx) = window.find(motif) {
        // Convert byte offset to char offset
        return window[..idx].chars().count();
    }

    // Center the motif in the window
    let window_chars = window.chars().count();
    let motif_chars = motif.chars().count();
    let center = window_chars / 2;
    let half = motif_chars / 2;
    center.saturating_sub(half)
}

/// Convert a char offset to a byte offset in a string.
fn char_offset_to_byte(s: &str, char_offset: usize) -> usize {
    s.char_indices()
        .nth(char_offset)
        .map(|(byte_idx, _)| byte_idx)
        .unwrap_or(s.len())
}

impl MotifMatcher {
    /// Create a new motif matcher.
    ///
    /// # Arguments
    /// * `motifs` - Map of {category: [motif_strings]}
    /// * `threshold` - Minimum match score (0-100)
    /// * `use_fuzzy` - Whether to use fuzzy matching for unmatched motifs
    pub fn new(
        motifs: HashMap<String, Vec<String>>,
        threshold: u32,
        use_fuzzy: bool,
    ) -> Result<Self, aho_corasick::BuildError> {
        let mut flat_motifs: Vec<(String, String)> = Vec::new();
        let mut patterns: Vec<String> = Vec::new();

        // Sort categories for deterministic order
        let mut sorted_cats: Vec<_> = motifs.keys().cloned().collect();
        sorted_cats.sort();

        for cat in &sorted_cats {
            for motif in &motifs[cat] {
                let lower = motif.to_lowercase();
                patterns.push(lower.clone());
                flat_motifs.push((lower, cat.clone()));
            }
        }

        let ac = AhoCorasick::new(&patterns)?;

        Ok(Self {
            ac,
            motifs: flat_motifs,
            threshold,
            use_fuzzy,
        })
    }

    /// Find all motif matches in text.
    ///
    /// Returns a list of `MotifMatch` structs sorted by position.
    pub fn find_matches(&self, text: &str, window_size: usize, step: usize) -> Vec<MotifMatch> {
        if text.is_empty() {
            return Vec::new();
        }

        let text_lower = text.to_lowercase();
        let mut results: Vec<MotifMatch> = Vec::new();
        let mut seen: HashSet<(usize, usize)> = HashSet::new(); // (position, motif_index)

        // Tier 1: Aho-Corasick exact matching
        let mut matched_motif_indices: HashSet<usize> = HashSet::new();

        for mat in self.ac.find_iter(&text_lower) {
            let motif_idx = mat.pattern().as_usize();
            let (ref motif, ref category) = self.motifs[motif_idx];
            let key = (mat.start(), motif_idx);

            if seen.insert(key) {
                matched_motif_indices.insert(motif_idx);
                results.push(MotifMatch {
                    motif: motif.clone(),
                    category: category.clone(),
                    position: mat.start(),
                    length: mat.end() - mat.start(),
                    score: 100.0,
                });
            }
        }

        // Tier 2: Fuzzy matching for unmatched motifs
        if self.use_fuzzy {
            let text_chars: Vec<char> = text_lower.chars().collect();
            let text_char_len = text_chars.len();

            for (motif_idx, (motif, category)) in self.motifs.iter().enumerate() {
                if matched_motif_indices.contains(&motif_idx) {
                    continue;
                }

                // Slide window across text
                let mut pos = 0usize;
                while pos < text_char_len {
                    let end = (pos + window_size).min(text_char_len);
                    let window: String = text_chars[pos..end].iter().collect();

                    if window.len() >= 10 {
                        let score = partial_ratio(motif, &window);
                        if score >= self.threshold {
                            // Find best position within window
                            let match_start = find_best_match_pos(&window, motif);
                            // Convert char offset to byte offset for the overall text
                            let abs_char_pos = pos + match_start;
                            let abs_byte_pos = char_offset_to_byte(&text_lower, abs_char_pos);
                            let match_byte_len = motif.len().min(window.len() - match_start);

                            let key = (abs_byte_pos, motif_idx);
                            if seen.insert(key) {
                                results.push(MotifMatch {
                                    motif: motif.clone(),
                                    category: category.clone(),
                                    position: abs_byte_pos,
                                    length: match_byte_len,
                                    score: score as f64,
                                });
                            }
                        }
                    }

                    pos += step;
                }
            }
        }

        // Sort by position
        results.sort_by_key(|m| m.position);

        results
    }

    /// Get merged match position spans.
    pub fn get_match_positions(&self, text: &str) -> Vec<(usize, usize)> {
        let matches = self.find_matches(text, 50, 25);

        if matches.is_empty() {
            return Vec::new();
        }

        let mut spans: Vec<(usize, usize)> = matches
            .iter()
            .map(|m| (m.position, m.position + m.length))
            .collect();
        spans.sort();

        // Merge overlapping
        let mut merged: Vec<(usize, usize)> = vec![spans[0]];
        for &(start, end) in &spans[1..] {
            let last = merged.last_mut().unwrap();
            if start <= last.1 {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_motifs() -> HashMap<String, Vec<String>> {
        let mut motifs = HashMap::new();
        motifs.insert(
            "instruction_override".into(),
            vec!["ignore previous".into(), "forget everything".into()],
        );
        motifs.insert(
            "role_injection".into(),
            vec!["you are now".into(), "pretend to be".into()],
        );
        motifs
    }

    #[test]
    fn test_exact_matching() {
        let matcher = MotifMatcher::new(test_motifs(), 75, false).unwrap();
        let matches = matcher.find_matches("please ignore previous instructions", 50, 25);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].category, "instruction_override");
        assert_eq!(matches[0].score, 100.0);
    }

    #[test]
    fn test_no_match() {
        let matcher = MotifMatcher::new(test_motifs(), 75, false).unwrap();
        let matches = matcher.find_matches("hello world, this is benign", 50, 25);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_get_match_positions() {
        let matcher = MotifMatcher::new(test_motifs(), 75, false).unwrap();
        let spans = matcher.get_match_positions("please ignore previous instructions");
        assert!(!spans.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let matcher = MotifMatcher::new(test_motifs(), 75, false).unwrap();
        let matches = matcher.find_matches("", 50, 25);
        assert!(matches.is_empty());
    }
}
