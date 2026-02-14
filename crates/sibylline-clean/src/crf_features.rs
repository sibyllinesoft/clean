use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::types::CrfTokenFeatures;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\S+").unwrap());
static SPECIAL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-zA-Z0-9\s]").unwrap());

static INSTRUCTION_KWS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
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
    ]
    .into_iter()
    .collect()
});

static ROLE_KWS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
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
    ]
    .into_iter()
    .collect()
});

static ACTION_KWS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
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
    ]
    .into_iter()
    .collect()
});

static JAILBREAK_KWS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
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
    ]
    .into_iter()
    .collect()
});

const MAX_TOKENS: usize = 2000;

/// Stamp all 4 keyword-category features for a context token into a feature map.
fn stamp_context_kws(map: &mut CrfTokenFeatures, prefix: &str, word: &str) {
    if INSTRUCTION_KWS.contains(word) {
        map.insert(format!("{}:kw:instr", prefix), "1".into());
    }
    if ROLE_KWS.contains(word) {
        map.insert(format!("{}:kw:role", prefix), "1".into());
    }
    if ACTION_KWS.contains(word) {
        map.insert(format!("{}:kw:act", prefix), "1".into());
    }
    if JAILBREAK_KWS.contains(word) {
        map.insert(format!("{}:kw:jail", prefix), "1".into());
    }
}

/// Check if a word is title case, matching Python's str.istitle().
///
/// A string is titlecased if uppercase characters follow uncased characters
/// and lowercase characters follow cased characters.
fn is_title_case(word: &str) -> bool {
    let mut cased_found = false;
    let mut previous_was_cased = false;

    for c in word.chars() {
        if c.is_uppercase() {
            if previous_was_cased {
                return false;
            }
            cased_found = true;
            previous_was_cased = true;
        } else if c.is_lowercase() {
            if !previous_was_cased {
                return false;
            }
            cased_found = true;
            previous_was_cased = true;
        } else {
            // Non-cased character (digit, punctuation, etc.)
            previous_was_cased = false;
        }
    }

    cased_found
}

/// Extract CRF features from text, returning a list of feature maps.
///
/// Each map has string keys and string values matching the Python _word_features output.
pub fn text_to_features(text: &str) -> Vec<CrfTokenFeatures> {
    let tokens: Vec<&str> = WORD_RE
        .find_iter(text)
        .take(MAX_TOKENS)
        .map(|m| m.as_str())
        .collect();

    let n = tokens.len();
    if n == 0 {
        return Vec::new();
    }

    // Pre-compute lowercased tokens
    let lower_tokens: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();

    let mut result: Vec<CrfTokenFeatures> = Vec::with_capacity(n);

    for i in 0..n {
        let word = tokens[i];
        let w = &lower_tokens[i];

        let mut map = CrfTokenFeatures::new();

        map.insert("bias".into(), "1".into());
        map.insert("w".into(), w.clone());

        // Suffix/prefix features — use char boundaries, not byte offsets
        let char_count = w.chars().count();

        // w[-3:] — last 3 characters
        if char_count >= 3 {
            let start = w.char_indices().nth(char_count - 3).unwrap().0;
            map.insert("w[-3:]".into(), w[start..].into());
        } else {
            map.insert("w[-3:]".into(), w.clone());
        }

        // w[-2:] — last 2 characters
        if char_count >= 2 {
            let start = w.char_indices().nth(char_count - 2).unwrap().0;
            map.insert("w[-2:]".into(), w[start..].into());
        } else {
            map.insert("w[-2:]".into(), w.clone());
        }

        // w[:3] — first 3 characters
        if char_count > 3 {
            let end = w.char_indices().nth(3).unwrap().0;
            map.insert("w[:3]".into(), w[..end].into());
        } else {
            // char_count <= 3: the whole string is the prefix
            map.insert("w[:3]".into(), w.clone());
        }

        map.insert("len".into(), format!("{}", char_count.min(20)));

        // Word shape features — match Python's str.isupper():
        // True if all cased chars are uppercase AND at least one cased char exists
        {
            let mut has_cased = false;
            let mut all_upper = true;
            for c in word.chars() {
                if c.is_uppercase() {
                    has_cased = true;
                } else if c.is_lowercase() {
                    all_upper = false;
                    break;
                }
                // Non-cased chars (digits, punct) are ignored
            }
            if has_cased && all_upper && word.len() > 1 {
                map.insert("ALLCAP".into(), "1".into());
            }
        }

        if is_title_case(word) {
            map.insert("Title".into(), "1".into());
        }

        if word.chars().all(|c| c.is_ascii_digit()) {
            map.insert("Digit".into(), "1".into());
        }

        if SPECIAL_RE.is_match(word) {
            map.insert("HasSpecial".into(), "1".into());
        }

        // Keyword features
        if INSTRUCTION_KWS.contains(w.as_str()) {
            map.insert("kw:instr".into(), "1".into());
        }
        if ROLE_KWS.contains(w.as_str()) {
            map.insert("kw:role".into(), "1".into());
        }
        if ACTION_KWS.contains(w.as_str()) {
            map.insert("kw:act".into(), "1".into());
        }
        if JAILBREAK_KWS.contains(w.as_str()) {
            map.insert("kw:jail".into(), "1".into());
        }

        // Position bucket (10 buckets)
        let pos_bucket = if n > 0 { (i * 10 / n).min(9) } else { 0 };
        map.insert("pos".into(), format!("{}", pos_bucket));

        // BOS / EOS (independent — not exclusive with context)
        if i == 0 {
            map.insert("BOS".into(), "1".into());
        }
        if i == n - 1 {
            map.insert("EOS".into(), "1".into());
        }

        // Context window: ±1 gets word + all 4 kw categories; ±2, ±3 get kw only
        for offset in [-3i32, -2, -1, 1, 2, 3] {
            let j = i as i32 + offset;
            if j < 0 || j >= n as i32 {
                continue;
            }
            let cw = &lower_tokens[j as usize];
            let prefix = if offset > 0 {
                format!("+{}", offset)
            } else {
                format!("{}", offset)
            };
            if offset == -1 || offset == 1 {
                map.insert(format!("{}:w", prefix), cw.clone());
            }
            stamp_context_kws(&mut map, &prefix, cw.as_str());
        }

        result.push(map);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        assert!(text_to_features("").is_empty());
    }

    #[test]
    fn test_single_word() {
        let features = text_to_features("hello");
        assert_eq!(features.len(), 1);
        assert_eq!(features[0]["w"], "hello");
        assert_eq!(features[0]["bias"], "1");
        assert_eq!(features[0]["BOS"], "1");
        assert_eq!(features[0]["EOS"], "1");
    }

    #[test]
    fn test_keyword_features() {
        let features = text_to_features("ignore previous instructions");
        assert_eq!(features.len(), 3);
        // "ignore" should have kw:instr
        assert_eq!(features[0].get("kw:instr"), Some(&"1".to_string()));
        // "previous" should have kw:instr
        assert_eq!(features[1].get("kw:instr"), Some(&"1".to_string()));
        // "instructions" should have kw:instr
        assert_eq!(features[2].get("kw:instr"), Some(&"1".to_string()));
    }

    #[test]
    fn test_context_window() {
        let features = text_to_features("the system is compromised");
        // "system" is at index 1, should have +1:w and -1:w context
        assert_eq!(features[1].get("+1:w"), Some(&"is".to_string()));
        assert_eq!(features[1].get("-1:w"), Some(&"the".to_string()));
    }

    #[test]
    fn test_title_case() {
        assert!(is_title_case("Hello"));
        assert!(is_title_case("Hello World"));
        assert!(!is_title_case("hello"));
        assert!(!is_title_case("HELLO"));
    }
}
