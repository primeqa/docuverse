"""Tests for docuverse.utils.text_tiler.TextTiler"""
import pytest
from transformers import AutoTokenizer
from docuverse.utils.text_tiler import TextTiler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bert_tokenizer():
    """Load BERT tokenizer once for the whole module."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def tiler_token(bert_tokenizer):
    """Token-based tiler (non-sentence-aligned), max 128 tokens."""
    return TextTiler(max_doc_length=128, stride=20, tokenizer=bert_tokenizer,
                     aligned_on_sentences=False)


@pytest.fixture(scope="module")
def tiler_char():
    """Character-based tiler, max 200 chars."""
    return TextTiler(max_doc_length=200, stride=30, tokenizer=None,
                     count_type='char', aligned_on_sentences=False)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_token_mode_sets_tokenizer(self, bert_tokenizer):
        t = TextTiler(max_doc_length=512, stride=50, tokenizer=bert_tokenizer,
                      aligned_on_sentences=False)
        assert t.tokenizer is bert_tokenizer
        assert t.count_type == TextTiler.COUNT_TYPE_TOKEN
        # max_doc_size should be reduced by special tokens
        assert t.max_doc_size == 512 - bert_tokenizer.num_special_tokens_to_add()

    def test_char_mode_no_tokenizer(self):
        t = TextTiler(max_doc_length=500, stride=50, tokenizer=None,
                      count_type='char', aligned_on_sentences=False)
        assert t.tokenizer is None
        assert t.count_type == TextTiler.COUNT_TYPE_CHAR
        assert t.max_doc_size == 500

    def test_string_tokenizer_name(self):
        t = TextTiler(max_doc_length=64, stride=10,
                      tokenizer='bert-base-uncased', aligned_on_sentences=False)
        assert t.tokenizer is not None

    def test_invalid_count_type_raises(self, bert_tokenizer):
        with pytest.raises(RuntimeError, match="count_type"):
            TextTiler(max_doc_length=64, stride=10, tokenizer=bert_tokenizer,
                      count_type='invalid')

    def test_invalid_tokenizer_type_raises(self):
        with pytest.raises(RuntimeError, match="tokenizer argument"):
            TextTiler(max_doc_length=64, stride=10, tokenizer=12345,
                      aligned_on_sentences=False)


# ---------------------------------------------------------------------------
# get_tokenized_length
# ---------------------------------------------------------------------------

class TestGetTokenizedLength:
    def test_basic_length(self, tiler_token):
        length = tiler_token.get_tokenized_length("hello world")
        assert isinstance(length, int)
        assert length > 0

    def test_empty_string(self, tiler_token):
        length = tiler_token.get_tokenized_length("")
        # Tokenizer returns special tokens even for empty string
        assert isinstance(length, int)

    def test_char_mode_returns_char_count(self, tiler_char):
        text = "hello world"
        assert tiler_char.get_tokenized_length(text) == len(text)

    def test_char_mode_forced_tok_returns_negative(self, tiler_char):
        # forced_tok=True but no tokenizer → -1
        assert tiler_char.get_tokenized_length("hello", forced_tok=True) == -1

    def test_longer_text_more_tokens(self, tiler_token):
        short = tiler_token.get_tokenized_length("hello")
        long = tiler_token.get_tokenized_length("hello " * 100)
        assert long > short


# ---------------------------------------------------------------------------
# cleanup_url
# ---------------------------------------------------------------------------

class TestCleanupUrl:
    def test_replaces_http_url(self):
        text = "Visit https://example.com/page for info."
        result = TextTiler.cleanup_url(text, normalize_text=False)
        assert "https://example.com" not in result
        assert "URL" in result

    def test_preserves_non_url_text(self):
        text = "No links here, just plain text."
        result = TextTiler.cleanup_url(text, normalize_text=False)
        assert result == text

    def test_normalize_text_flag(self):
        # \xa0 is a non-breaking space; NFKC normalizes it to a regular space
        text = "before\xa0https://example.com after"
        result = TextTiler.cleanup_url(text, normalize_text=True)
        assert "URL" in result


# ---------------------------------------------------------------------------
# create_tiles — short text (no splitting needed)
# ---------------------------------------------------------------------------

class TestCreateTilesShort:
    def test_short_text_single_tile(self, tiler_token):
        tiles = tiler_token.create_tiles("doc1", "Hello world.", title="Greetings")
        assert len(tiles) == 1
        assert tiles[0]['id'].startswith("doc1")
        assert "Hello world." in tiles[0]['text']

    def test_none_text_returns_empty(self, tiler_token):
        assert tiler_token.create_tiles("doc1", None) == []

    def test_title_prepended_when_not_in_text(self, tiler_token):
        tiles = tiler_token.create_tiles("doc1", "Some body text.", title="My Title")
        assert tiles[0]['text'].startswith("My Title")

    def test_title_not_duplicated_when_in_text(self, tiler_token):
        tiles = tiler_token.create_tiles("doc1", "My Title\nSome body.", title="My Title")
        # Title is already at position 0 in text, so it shouldn't be prepended again
        assert not tiles[0]['text'].startswith("My Title\nMy Title")

    def test_template_fields_preserved(self, tiler_token):
        tiles = tiler_token.create_tiles("doc1", "Short text.", title="T",
                                         template={"source": "test"})
        assert tiles[0]['source'] == "test"

    def test_char_mode_single_tile(self, tiler_char):
        tiles = tiler_char.create_tiles("doc1", "Hello.", title="T")
        assert len(tiles) == 1


# ---------------------------------------------------------------------------
# create_tiles — long text (splitting)
# ---------------------------------------------------------------------------

class TestCreateTilesLong:
    def test_long_text_produces_multiple_tiles(self, tiler_token):
        long_text = "The quick brown fox jumps over the lazy dog. " * 200
        tiles = tiler_token.create_tiles("doc1", long_text, title="Title")
        assert len(tiles) > 1

    def test_tile_ids_are_unique(self, tiler_token):
        long_text = "Word " * 500
        tiles = tiler_token.create_tiles("doc1", long_text, title="T")
        ids = [t['id'] for t in tiles]
        assert len(ids) == len(set(ids))

    def test_all_tiles_have_text(self, tiler_token):
        long_text = "Sentence here. " * 300
        tiles = tiler_token.create_tiles("doc1", long_text, title="T")
        for tile in tiles:
            assert len(tile['text'].strip()) > 0

    def test_url_replaced_in_tiles(self, tiler_token):
        text = "Go to https://example.com/page now. " * 200
        tiles = tiler_token.create_tiles("doc1", text, title="T", remove_url=True)
        for tile in tiles:
            assert "https://example.com" not in tile['text']

    def test_title_handling_none(self, tiler_token):
        long_text = "Some words here. " * 300
        tiles = tiler_token.create_tiles("doc1", long_text, title="Title",
                                         title_handling="none")
        for tile in tiles:
            assert not tile['text'].startswith("Title\n")


# ---------------------------------------------------------------------------
# compute_intervals (pure function)
# ---------------------------------------------------------------------------

class TestComputeIntervals:
    def test_single_segment_fits(self):
        result = TextTiler.compute_intervals(
            segment_lengths=[10], max_length=100, first_length=100, stride=5)
        assert result == [[0, 0]]

    def test_two_segments_fit_in_one(self):
        result = TextTiler.compute_intervals(
            segment_lengths=[10, 10], max_length=100, first_length=100, stride=5)
        assert result == [[0, 1]]

    def test_segments_require_split(self):
        # 4 segments of 30 each, max 50 → need multiple intervals
        result = TextTiler.compute_intervals(
            segment_lengths=[30, 30, 30, 30], max_length=50, first_length=50, stride=10)
        assert len(result) >= 2
        # First interval starts at 0
        assert result[0][0] == 0
        # Last interval ends at last segment
        assert result[-1][1] == 3

    def test_first_length_smaller(self):
        # first_length=20 can only fit first segment, rest use max_length=60
        result = TextTiler.compute_intervals(
            segment_lengths=[15, 15, 15, 15], max_length=60, first_length=20, stride=5)
        assert len(result) >= 2
        assert result[0] == [0, 0]  # Only first segment in first interval

    def test_all_fit_in_one_interval(self):
        result = TextTiler.compute_intervals(
            segment_lengths=[5, 5, 5, 5], max_length=100, first_length=100, stride=5)
        assert len(result) == 1
        assert result[0] == [0, 3]

    def test_many_small_segments(self):
        # 20 segments of size 10, max 50 → ~4+ intervals
        lengths = [10] * 20
        result = TextTiler.compute_intervals(
            segment_lengths=lengths, max_length=50, first_length=50, stride=10)
        assert len(result) >= 4
        # Covers all segments
        assert result[-1][1] == 19


# ---------------------------------------------------------------------------
# _need_to_add_title
# ---------------------------------------------------------------------------

class TestNeedToAddTitle:
    def test_first_tile_all(self):
        assert TextTiler._need_to_add_title(0, 'all', False) is True

    def test_first_tile_first(self):
        assert TextTiler._need_to_add_title(0, 'first', False) is True

    def test_first_tile_none(self):
        assert TextTiler._need_to_add_title(0, 'none', False) is False

    def test_first_tile_title_in_text(self):
        assert TextTiler._need_to_add_title(0, 'all', True) is False

    def test_later_tile_all(self):
        assert TextTiler._need_to_add_title(1, 'all', False) is True

    def test_later_tile_first(self):
        assert TextTiler._need_to_add_title(1, 'first', False) is False

    def test_later_tile_none(self):
        assert TextTiler._need_to_add_title(5, 'none', False) is False


# ---------------------------------------------------------------------------
# split_text — non-sentence token mode
# ---------------------------------------------------------------------------

class TestSplitTextToken:
    def test_short_text_returns_single(self, tiler_token):
        texts, positions, added = tiler_token.split_text(
            text="Short text.", tokenizer=tiler_token.tokenizer,
            title="T", max_length=500, stride=20, title_handling='all')
        assert len(texts) == 1
        assert positions[0] == [0, len("Short text.")]

    def test_long_text_produces_multiple(self, tiler_token):
        long_text = "The quick brown fox. " * 200
        texts, positions, added = tiler_token.split_text(
            text=long_text, tokenizer=tiler_token.tokenizer,
            title="T", max_length=100, stride=20, title_handling='none')
        assert len(texts) > 1
        assert len(texts) == len(positions) == len(added)

    def test_positions_are_valid_ranges(self, tiler_token):
        long_text = "Word " * 500
        texts, positions, added = tiler_token.split_text(
            text=long_text, tokenizer=tiler_token.tokenizer,
            title="", max_length=100, stride=20, title_handling='none')
        for start, end in positions:
            assert 0 <= start < end


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_title(self, tiler_token):
        tiles = tiler_token.create_tiles("doc1", "Some text here.", title="")
        assert len(tiles) >= 1

    def test_very_long_single_word(self, tiler_token):
        # A single very long "word" — tokenizer will subword-split it
        text = "supercalifragilisticexpialidocious" * 50
        tiles = tiler_token.create_tiles("doc1", text, title="T")
        assert len(tiles) >= 1

    def test_unicode_text(self, tiler_token):
        text = "Héllo wörld! Ñoño está aquí. 日本語テスト。" * 20
        tiles = tiler_token.create_tiles("doc1", text, title="Unicode")
        assert len(tiles) >= 1
        for tile in tiles:
            assert len(tile['text']) > 0