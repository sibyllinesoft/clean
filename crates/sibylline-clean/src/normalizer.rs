use unicode_normalization::UnicodeNormalization;

/// Check if a character is a zero-width or invisible character.
#[inline]
fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{200b}' | // Zero-width space
        '\u{200c}' | // Zero-width non-joiner
        '\u{200d}' | // Zero-width joiner
        '\u{2060}' | // Word joiner
        '\u{2061}' | // Function application
        '\u{2062}' | // Invisible times
        '\u{2063}' | // Invisible separator
        '\u{2064}' | // Invisible plus
        '\u{feff}' | // BOM / zero-width no-break space
        '\u{00ad}' | // Soft hyphen
        '\u{034f}' | // Combining grapheme joiner
        '\u{061c}' | // Arabic letter mark
        '\u{115f}' | // Hangul choseong filler
        '\u{1160}' | // Hangul jungseong filler
        '\u{17b4}' | // Khmer vowel inherent aq
        '\u{17b5}' | // Khmer vowel inherent aa
        '\u{180e}' | // Mongolian vowel separator
        '\u{3164}' | // Hangul filler
        '\u{ffa0}' // Halfwidth hangul filler
    )
}

/// Check if a character is a bidirectional override character.
#[inline]
fn is_bidi(c: char) -> bool {
    matches!(
        c,
        '\u{202a}' | // Left-to-right embedding
        '\u{202b}' | // Right-to-left embedding
        '\u{202c}' | // Pop directional formatting
        '\u{202d}' | // Left-to-right override
        '\u{202e}' | // Right-to-left override
        '\u{2066}' | // Left-to-right isolate
        '\u{2067}' | // Right-to-left isolate
        '\u{2068}' | // First strong isolate
        '\u{2069}' // Pop directional isolate
    )
}

/// Fused text normalization: NFKC + strip invisible + collapse whitespace + lowercase.
///
/// Equivalent to TextNormalizer.normalize() when confusables are disabled.
pub fn normalize_text(text: &str) -> String {
    // Step 1: NFKC normalization
    let nfkc: String = text.nfkc().collect();

    // Steps 2-4 in a single pass: strip invisible/bidi, collapse whitespace, lowercase
    let mut result = String::with_capacity(nfkc.len());
    let mut last_was_space = true; // Start true to strip leading whitespace

    for c in nfkc.chars() {
        // Skip invisible and bidi characters
        if is_invisible(c) || is_bidi(c) {
            continue;
        }

        if c.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            for lc in c.to_lowercase() {
                result.push(lc);
            }
            last_was_space = false;
        }
    }

    // Strip trailing space
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_normalization() {
        assert_eq!(normalize_text("Hello World"), "hello world");
    }

    #[test]
    fn test_fullwidth() {
        // Fullwidth "Hello" -> "hello"
        assert_eq!(
            normalize_text("\u{ff28}\u{ff45}\u{ff4c}\u{ff4c}\u{ff4f}"),
            "hello"
        );
    }

    #[test]
    fn test_zero_width_stripping() {
        assert_eq!(normalize_text("he\u{200b}llo"), "hello");
    }

    #[test]
    fn test_bidi_stripping() {
        assert_eq!(normalize_text("he\u{202a}llo"), "hello");
    }

    #[test]
    fn test_whitespace_collapse() {
        assert_eq!(normalize_text("  hello   world  "), "hello world");
    }

    #[test]
    fn test_empty() {
        assert_eq!(normalize_text(""), "");
    }
}
