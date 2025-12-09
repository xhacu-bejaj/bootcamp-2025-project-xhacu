"""
Text chunking strategies for breaking documents into smaller pieces.

This module provides abstract interfaces and concrete implementations
for chunking text documents while respecting maximum chunk lengths.
"""

from abc import ABC, abstractmethod
import re
from typing import List

from app.core.config import settings


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies."""

    def __init__(self, max_length: int | None = None):
        """
        Initialize the chunking strategy.

        Args:
            max_length: Maximum length for each chunk. Defaults to configured MAX_CHUNK_LENGTH.
        """
        self.max_length = max_length or settings.MAX_CHUNK_LENGTH

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks according to the strategy.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks

        Raises:
            ValueError: If a single sentence/phrase exceeds max_length
        """
        ...

    def _validate_chunk(self, chunk: str) -> None:
        """
        Validate that a chunk doesn't exceed max length.

        Args:
            chunk: The chunk to validate

        Raises:
            ValueError: If chunk exceeds max_length
        """
        if len(chunk) > self.max_length:
            raise ValueError(
                f"Chunk exceeds maximum length of {self.max_length} characters. "
                f"Got {len(chunk)} characters."
            )


class PhraseChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that splits text by sentences/phrases.

    This strategy:
    - Splits text into sentences using common sentence terminators
    - Handles titles and section headers specially
    - Groups sentences to respect max_length
    - Preserves paragraph boundaries when possible
    """

    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    TITLE_PATTERN = re.compile(r'^[A-Z][^\n]{0,80}(?<![.!?])$', re.MULTILINE)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks by phrases/sentences.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks, each within max_length

        Raises:
            ValueError: If a single sentence exceeds max_length
        """
        if not text or not text.strip():
            return []

        chunks: List[str] = []
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if self._is_title(paragraph):
                if len(paragraph) <= self.max_length:
                    chunks.append(paragraph)
                else:
                    chunks.extend(self._split_long_text(paragraph))
                continue

            sentences = self._split_into_sentences(paragraph)

            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(sentence) > self.max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.extend(self._split_long_text(sentence))
                    continue

                test_chunk = f"{current_chunk} {sentence}".strip()
                if len(test_chunk) <= self.max_length:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

        return chunks

    def _is_title(self, text: str) -> bool:
        """
        Determine if text appears to be a title or header.

        Args:
            text: Text to check

        Returns:
            True if text looks like a title/header
        """
        
        lines = text.split('\n')
        if len(lines) != 1:
            return False

        text = text.strip()
        if len(text) > 100: 
            return False

        if not text[0].isupper():
            return False

        if text.endswith(('.', '!', '?')):
            return False

        return True

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_text(self, text: str) -> List[str]:
        """
        Split text that exceeds max_length into smaller chunks.

        Falls back to splitting by words or characters when necessary.

        Args:
            text: Text to split

        Returns:
            List of chunks

        Raises:
            ValueError: If text cannot be chunked (e.g., single word too long)
        """
        chunks: List[str] = []
        words = text.split()

        if not words:
            return []

        current_chunk = ""
        for word in words:
            if len(word) > self.max_length:
                raise ValueError(
                    f"Cannot chunk text: single word exceeds maximum length. "
                    f"Word length: {len(word)}, max length: {self.max_length}"
                )

            test_chunk = f"{current_chunk} {word}".strip()
            if len(test_chunk) <= self.max_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class ChunkingStrategyFactory:
    """Factory for creating chunking strategy instances."""

    _strategies: dict[str, type[ChunkingStrategy]] = {
        "phrase": PhraseChunkingStrategy,
    }

    @classmethod
    def get_strategy(cls, strategy_name: str, max_length: int | None = None) -> ChunkingStrategy:
        """
        Get a chunking strategy instance by name.

        Args:
            strategy_name: Name of the strategy to create
            max_length: Maximum chunk length (optional)

        Returns:
            ChunkingStrategy instance

        Raises:
            ValueError: If strategy_name is not registered
        """
        strategy_class = cls._strategies.get(strategy_name.lower())
        if not strategy_class:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown chunking strategy: '{strategy_name}'. "
                f"Available strategies: {available}"
            )

        return strategy_class(max_length=max_length)

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[ChunkingStrategy]) -> None:
        """
        Register a new chunking strategy.

        Args:
            name: Name for the strategy
            strategy_class: ChunkingStrategy class to register
        """
        cls._strategies[name.lower()] = strategy_class

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Get list of available strategy names.

        Returns:
            List of registered strategy names
        """
        return list(cls._strategies.keys())
