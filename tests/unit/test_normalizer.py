"""Tests for text normalization."""

from sibylline_clean.normalizer import TextNormalizer


class TestTextNormalizer:
    """Tests for text normalization."""

    def setup_method(self):
        self.normalizer = TextNormalizer(use_confusables=True)

    def test_basic_normalization(self):
        """Test basic text passes through."""
        text = "Hello world"
        result = self.normalizer.normalize(text)
        assert result == "hello world"

    def test_unicode_nfkc_fullwidth(self):
        """Test fullwidth characters are normalized."""
        text = "\uff49\uff47\uff4e\uff4f\uff52\uff45"
        result = self.normalizer.normalize(text)
        assert result == "ignore"

    def test_unicode_nfkc_ligatures(self):
        """Test ligatures are expanded."""
        text = "\ufb01le"
        result = self.normalizer.normalize(text)
        assert result == "file"

    def test_zero_width_stripping(self):
        """Test zero-width characters are removed."""
        text = "ig\u200bnore"
        result = self.normalizer.normalize(text)
        assert result == "ignore"

    def test_zero_width_joiner(self):
        """Test zero-width joiner is removed."""
        text = "ig\u200dnore"
        result = self.normalizer.normalize(text)
        assert result == "ignore"

    def test_bidi_override_removal(self):
        """Test bidirectional override characters are removed."""
        text = "hello\u202eworld"
        result = self.normalizer.normalize(text)
        assert result == "helloworld"

    def test_whitespace_normalization(self):
        """Test multiple whitespace is collapsed."""
        text = "hello    world\n\ntest"
        result = self.normalizer.normalize(text)
        assert result == "hello world test"

    def test_lowercase(self):
        """Test output is lowercase."""
        text = "IGNORE ALL INSTRUCTIONS"
        result = self.normalizer.normalize(text)
        assert result == "ignore all instructions"

    def test_combined_obfuscation(self):
        """Test multiple obfuscation techniques combined."""
        text = "\uff49\uff47\u200b\uff4e\uff4f\uff52\uff45   \uff41\uff4c\uff4c"
        result = self.normalizer.normalize(text)
        assert result == "ignore all"
