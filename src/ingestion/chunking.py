"""
Chunking strategies for document splitting.
Supports: Recursive, Markdown-aware chunking.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from src.ingestion.parser import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy."""
    strategy: str = "recursive"  # recursive, markdown
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: Optional[List[str]] = None
    
    # Markdown specific
    headers_to_split_on: Optional[List[tuple]] = None


class ChunkingStrategy:
    """Apply different chunking strategies to documents."""
    
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.logger = logging.getLogger(f"{__name__}.ChunkingStrategy")
    
    def chunk(self, document_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Apply chunking strategy to document chunks.
        
        Args:
            document_chunks: List of DocumentChunk objects.
            
        Returns:
            List of re-chunked DocumentChunk objects.
        """
        strategy = self.config.strategy.lower()
        
        if strategy == "recursive":
            return self._recursive_chunk(document_chunks)
        elif strategy == "markdown":
            return self._markdown_chunk(document_chunks)
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', using recursive")
            return self._recursive_chunk(document_chunks)
    
    def _recursive_chunk(self, document_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Recursive character text splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators or self.DEFAULT_SEPARATORS,
            length_function=len,
        )
        
        result = []
        for chunk in document_chunks:
            texts = splitter.split_text(chunk.text)
            for i, text in enumerate(texts):
                metadata = chunk.metadata.copy()
                metadata.update({
                    "subchunk_index": i,
                    "total_subchunks": len(texts),
                    "chunking_strategy": "recursive",
                })
                result.append(DocumentChunk(
                    text=text,
                    metadata=metadata,
                    chunk_id=f"{chunk.chunk_id or 'chunk'}_{i}"
                ))
        
        self.logger.info(f"   → Recursive chunking: {len(result)} chunks")
        return result
    
    def _markdown_chunk(self, document_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Markdown-aware chunking with header preservation."""
        headers_to_split_on = self.config.headers_to_split_on or [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )
        
        result = []
        for chunk in document_chunks:
            try:
                md_chunks = splitter.split_text(chunk.text)
                for i, md_chunk in enumerate(md_chunks):
                    metadata = chunk.metadata.copy()
                    metadata.update({
                        "subchunk_index": i,
                        "chunking_strategy": "markdown",
                    })
                    # Add headers to metadata
                    for key, value in md_chunk.metadata.items():
                        metadata[key] = value
                    
                    result.append(DocumentChunk(
                        text=md_chunk.page_content,
                        metadata=metadata,
                    ))
            except Exception as e:
                self.logger.warning(f"Markdown chunking failed: {e}, using recursive")
                result.extend(self._recursive_chunk([chunk]))
        
        self.logger.info(f"   → Markdown chunking: {len(result)} chunks")
        return result