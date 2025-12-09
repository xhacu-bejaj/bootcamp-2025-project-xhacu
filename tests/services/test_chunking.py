"""
Tests for the chunking service.

This module tests text chunking strategies for breaking documents
into smaller pieces while respecting maximum chunk lengths.
"""

import pytest
from app.services.chunking import (
    ChunkingStrategy,
    PhraseChunkingStrategy,
    ChunkingStrategyFactory
)


class TestPhraseChunkingStrategy:
    """Tests for PhraseChunkingStrategy."""

    def test_chunk_simple_text(self):
        """Test chunking simple text with multiple sentences."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "This is sentence one. This is sentence two. This is sentence three."
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)
        # All text should be preserved
        assert "sentence one" in " ".join(chunks)
        assert "sentence two" in " ".join(chunks)
        assert "sentence three" in " ".join(chunks)

    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        strategy = PhraseChunkingStrategy(max_length=100)
        
        assert strategy.chunk_text("") == []
        assert strategy.chunk_text("   ") == []
        assert strategy.chunk_text("\n\n") == []

    def test_chunk_single_sentence_within_limit(self):
        """Test chunking single sentence within max_length."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "This is a short sentence."
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_respects_max_length(self):
        """Test that all chunks respect max_length."""
        strategy = PhraseChunkingStrategy(max_length=50)
        text = "Short sentence. Another short one. And one more. Final sentence here."
        
        chunks = strategy.chunk_text(text)
        
        assert all(len(chunk) <= 50 for chunk in chunks)
        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_chunk_long_sentence_exceeds_limit(self):
        """Test that single sentence exceeding max_length is split by words."""
        strategy = PhraseChunkingStrategy(max_length=30)
        text = "This is a very long sentence that definitely exceeds the maximum length."
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 30 for chunk in chunks)
        # All words should be preserved
        original_words = set(text.split())
        chunk_words = set(" ".join(chunks).split())
        assert original_words == chunk_words

    def test_chunk_single_word_too_long_raises_error(self):
        """Test that single word exceeding max_length raises ValueError."""
        strategy = PhraseChunkingStrategy(max_length=10)
        text = "This verylongwordthatexceedslimit here."
        
        with pytest.raises(ValueError, match="single word exceeds maximum length"):
            strategy.chunk_text(text)

    def test_chunk_preserves_paragraph_boundaries(self):
        """Test that paragraph boundaries are respected."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "Paragraph one sentence.\n\nParagraph two sentence."
        
        chunks = strategy.chunk_text(text)
        
        # Each paragraph should be in separate chunks if possible
        assert len(chunks) >= 2
        assert "Paragraph one" in chunks[0]
        assert "Paragraph two" in chunks[1]

    def test_chunk_handles_multiple_paragraphs(self):
        """Test chunking text with multiple paragraphs."""
        strategy = PhraseChunkingStrategy(max_length=200)
        text = """First paragraph with some text.

Second paragraph with more text.

Third paragraph with final text."""
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        combined = " ".join(chunks)
        assert "First paragraph" in combined
        assert "Second paragraph" in combined
        assert "Third paragraph" in combined

    def test_is_title_detects_titles(self):
        """Test title detection logic."""
        strategy = PhraseChunkingStrategy(max_length=100)
        
        # Should be detected as titles
        assert strategy._is_title("Introduction")
        assert strategy._is_title("Chapter One")
        assert strategy._is_title("Section 2: Methods")
        
        # Should NOT be detected as titles
        assert not strategy._is_title("This is a sentence.")
        assert not strategy._is_title("this is lowercase")
        assert not strategy._is_title("Multiple\nlines\nhere")
        assert not strategy._is_title("A" * 150)  # Too long

    def test_chunk_handles_titles(self):
        """Test that titles are handled specially."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = """Introduction

This is the first paragraph with content.

Methods

This is the methods section with details."""
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        # Titles should be preserved
        combined = " ".join(chunks)
        assert "Introduction" in combined
        assert "Methods" in combined

    def test_chunk_long_title_is_split(self):
        """Test that very long titles exceeding max_length are split."""
        strategy = PhraseChunkingStrategy(max_length=30)
        text = "Very Long Title That Definitely Exceeds The Maximum Length Limit"
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 30 for chunk in chunks)

    def test_split_into_sentences(self):
        """Test sentence splitting logic."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "First sentence. Second sentence! Third sentence? Fourth."
        
        sentences = strategy._split_into_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_split_into_sentences_handles_edge_cases(self):
        """Test sentence splitting with edge cases."""
        strategy = PhraseChunkingStrategy(max_length=100)
        
        # No sentence terminators
        sentences = strategy._split_into_sentences("Just one phrase")
        assert len(sentences) == 1
        
        # Multiple spaces
        sentences = strategy._split_into_sentences("One.   Two.   Three.")
        assert len(sentences) == 3

    def test_split_long_text_by_words(self):
        """Test splitting long text by words."""
        strategy = PhraseChunkingStrategy(max_length=20)
        text = "This is a long text that needs splitting"
        
        chunks = strategy._split_long_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)
        # Verify all words preserved
        assert set(text.split()) == set(" ".join(chunks).split())

    def test_split_long_text_empty_returns_empty(self):
        """Test splitting empty text returns empty list."""
        strategy = PhraseChunkingStrategy(max_length=50)
        
        chunks = strategy._split_long_text("")
        
        assert chunks == []

    def test_chunk_with_special_characters(self):
        """Test chunking text with special characters."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "This has special chars: @#$%! And more? Yes."
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        assert "@#$%!" in " ".join(chunks)

    def test_chunk_with_numbers(self):
        """Test chunking text with numbers."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "The year 2025 is here. We have 123 items. Total is 456.789."
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        combined = " ".join(chunks)
        assert "2025" in combined
        assert "123" in combined
        assert "456.789" in combined

    def test_chunk_with_unicode(self):
        """Test chunking text with unicode characters."""
        strategy = PhraseChunkingStrategy(max_length=100)
        text = "Hello world. Hola mundo. Bonjour le monde. 你好世界."
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        combined = " ".join(chunks)
        assert "Hola mundo" in combined
        assert "你好世界" in combined

    def test_validate_chunk_accepts_valid(self):
        """Test chunk validation accepts valid chunks."""
        strategy = PhraseChunkingStrategy(max_length=50)
        
        # Should not raise
        strategy._validate_chunk("This is a valid chunk")
        strategy._validate_chunk("A" * 50)  # Exactly at limit

    def test_validate_chunk_rejects_too_long(self):
        """Test chunk validation rejects chunks exceeding max_length."""
        strategy = PhraseChunkingStrategy(max_length=50)
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            strategy._validate_chunk("A" * 51)


class TestChunkingStrategyFactory:
    """Tests for ChunkingStrategyFactory."""

    def test_get_phrase_strategy(self):
        """Test getting phrase chunking strategy."""
        strategy = ChunkingStrategyFactory.get_strategy("phrase")
        
        assert isinstance(strategy, PhraseChunkingStrategy)

    def test_get_strategy_case_insensitive(self):
        """Test strategy name is case insensitive."""
        strategy1 = ChunkingStrategyFactory.get_strategy("phrase")
        strategy2 = ChunkingStrategyFactory.get_strategy("PHRASE")
        strategy3 = ChunkingStrategyFactory.get_strategy("Phrase")
        
        assert type(strategy1) == type(strategy2) == type(strategy3)

    def test_get_strategy_with_custom_max_length(self):
        """Test creating strategy with custom max_length."""
        strategy = ChunkingStrategyFactory.get_strategy("phrase", max_length=200)
        
        assert strategy.max_length == 200

    def test_get_strategy_invalid_name_raises_error(self):
        """Test invalid strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkingStrategyFactory.get_strategy("invalid_strategy")

    def test_get_strategy_invalid_name_shows_available(self):
        """Test error message shows available strategies."""
        try:
            ChunkingStrategyFactory.get_strategy("invalid")
        except ValueError as e:
            assert "Available strategies" in str(e)
            assert "phrase" in str(e)

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = ChunkingStrategyFactory.list_strategies()
        
        assert "phrase" in strategies
        assert isinstance(strategies, list)

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        class CustomStrategy(ChunkingStrategy):
            def chunk_text(self, text: str):
                return [text]
        
        ChunkingStrategyFactory.register_strategy("custom", CustomStrategy)
        
        # Should be able to get the custom strategy
        strategy = ChunkingStrategyFactory.get_strategy("custom")
        assert isinstance(strategy, CustomStrategy)
        
        # Should appear in list
        assert "custom" in ChunkingStrategyFactory.list_strategies()

    def test_register_strategy_overwrites_existing(self):
        """Test registering strategy with existing name overwrites it."""
        class NewPhraseStrategy(ChunkingStrategy):
            def chunk_text(self, text: str):
                return ["new implementation"]
        
        original_count = len(ChunkingStrategyFactory.list_strategies())
        ChunkingStrategyFactory.register_strategy("phrase", NewPhraseStrategy)
        
        # Should not increase count (overwrote existing)
        assert len(ChunkingStrategyFactory.list_strategies()) == original_count
        
        # Should return new implementation
        strategy = ChunkingStrategyFactory.get_strategy("phrase")
        assert isinstance(strategy, NewPhraseStrategy)
        
        # Restore original for other tests
        ChunkingStrategyFactory.register_strategy("phrase", PhraseChunkingStrategy)


class TestChunkingStrategyIntegration:
    """Integration tests for chunking strategies."""

    def test_chunk_realistic_document(self):
        """Test chunking a realistic document."""
        strategy = PhraseChunkingStrategy(max_length=200)
        text = """Introduction

This document describes the chunking functionality. It should handle various types of text.

Implementation Details

The implementation uses regular expressions for sentence detection. It also handles edge cases like very long sentences.

Conclusion

The chunking strategy works well for most documents."""
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 200 for chunk in chunks)
        
        # Verify content preservation
        combined = " ".join(chunks)
        assert "Introduction" in combined
        assert "Implementation Details" in combined
        assert "Conclusion" in combined

    def test_chunk_and_reassemble(self):
        """Test that chunked text can be reassembled."""
        strategy = PhraseChunkingStrategy(max_length=100)
        original = "First sentence here. Second sentence. Third one. Fourth sentence."
        
        chunks = strategy.chunk_text(original)
        reassembled = " ".join(chunks)
        
        # All words should be preserved
        original_words = set(original.split())
        reassembled_words = set(reassembled.split())
        assert original_words == reassembled_words

    def test_chunk_article_like_text(self):
        """Test chunking article-like text with multiple sections."""
        strategy = PhraseChunkingStrategy(max_length=150)
        text = """Breaking News

Scientists have made a breakthrough discovery. The research team published their findings yesterday.

Study Details

The study involved analyzing data from multiple sources. Results showed significant improvements in efficiency."""
        
        chunks = strategy.chunk_text(text)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 150 for chunk in chunks)

    def test_different_max_lengths_produce_different_chunking(self):
        """Test that different max_length values produce different chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        
        strategy_small = PhraseChunkingStrategy(max_length=30)
        strategy_large = PhraseChunkingStrategy(max_length=200)
        
        chunks_small = strategy_small.chunk_text(text)
        chunks_large = strategy_large.chunk_text(text)
        
        # Smaller max_length should produce more chunks
        assert len(chunks_small) > len(chunks_large)
