"""
Document parser — simplified version for text files.
Supports: TXT, MD (PDF and DOCX support is optional).
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None


class DocumentParser:
    """
    Parser for document files.
    Currently supports: .txt, .md
    PDF and DOCX support requires additional dependencies.
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md"}
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DocumentParser")
    
    def parse(self, file_path: Path) -> List[DocumentChunk]:
        """
        Parse a document and return chunks with metadata.
        
        Args:
            file_path: Path to the document.
            
        Returns:
            List of DocumentChunk objects.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            self.logger.warning(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
            # Try to parse as text anyway
            return self._parse_text(file_path)
        
        self.logger.info(f"Parsing: {file_path.name} ({extension})")
        
        if extension in (".txt", ".md"):
            return self._parse_text(file_path)
        
        return self._parse_text(file_path)
    
    def _parse_text(self, file_path: Path) -> List[DocumentChunk]:
        """Parse plain text and markdown files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, "r", encoding="cp1251") as f:
                text = f.read()
        
        # Split by double newlines for basic structure
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        if not paragraphs:
            # If no paragraphs, split by single newlines
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            # Detect if it's a header (starts with # or ##)
            is_header = paragraph.startswith("#") or paragraph.startswith("##")
            
            chunks.append(DocumentChunk(
                text=paragraph,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "chunk_index": i,
                    "total_chunks": len(paragraphs),
                    "file_type": file_path.suffix.lower().replace(".", ""),
                    "is_header": is_header,
                }
            ))
        
        self.logger.info(f"   → {len(chunks)} chunks extracted")
        return chunks
    
    def parse_directory(self, directory: Path) -> Dict[str, List[DocumentChunk]]:
        """
        Parse all supported documents in a directory.
        
        Args:
            directory: Path to directory containing documents.
            
        Returns:
            Dictionary mapping filename to list of chunks.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = {}
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    chunks = self.parse(file_path)
                    if chunks:
                        results[file_path.name] = chunks
                        self.logger.info(f"✅ {file_path.name}: {len(chunks)} chunks")
                except Exception as e:
                    self.logger.error(f"❌ {file_path.name}: {e}")
        
        return results