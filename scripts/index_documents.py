"""
Script for indexing documents into Qdrant.
"""

import sys
import logging
from pathlib import Path

# Добавляем корневую папку проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion import DocumentParser, ChunkingStrategy, EmbeddingFactory
from src.retrieval import QdrantClientWrapper
from src.ingestion.chunking import ChunkingConfig

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def index_documents():
    """Index all documents from data/raw/ into Qdrant."""
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Document Indexing Pipeline")
    logger.info("=" * 60)
    logger.info(f"📌 Environment: {settings.environment}")
    logger.info(f"📌 Qdrant URL: {settings.qdrant_url}")
    logger.info(f"📌 Collection: {settings.qdrant_collection}")
    
    # 1. Initialize components
    parser = DocumentParser()
    chunk_config = ChunkingConfig(
        strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
    )
    chunker = ChunkingStrategy(config=chunk_config)
    embedder = EmbeddingFactory()
    qdrant = QdrantClientWrapper()
    
    # 2. Create collection in Qdrant
    logger.info("📁 Creating/verifying Qdrant collection...")
    qdrant.create_collection(force=False)
    
    # 3. Check raw directory
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        logger.error("❌ Directory 'data/raw/' not found! Create it and add documents.")
        return
    
    # 4. Parse all documents
    logger.info(f"📄 Scanning: {raw_dir}")
    parsed_docs = parser.parse_directory(raw_dir)
    
    if not parsed_docs:
        logger.warning("⚠️ No documents found in data/raw/")
        return
    
    total_files = len(parsed_docs)
    total_chunks = sum(len(chunks) for chunks in parsed_docs.values())
    logger.info(f"📄 Found {total_files} files, {total_chunks} initial chunks")
    
    # 5. Process each document
    all_texts = []
    all_metadata = []
    
    for filename, chunks in parsed_docs.items():
        logger.info(f"📄 Processing: {filename} ({len(chunks)} chunks)")
        
        # Chunking
        try:
            chunked = chunker.chunk(chunks)
            logger.info(f"   → After chunking: {len(chunked)} chunks")
        except Exception as e:
            logger.error(f"   ❌ Chunking failed: {e}")
            continue
        
        # Collect texts and metadata
        for chunk in chunked:
            all_texts.append(chunk.text)
            all_metadata.append({
                "source": filename,
                **chunk.metadata,
            })
    
    if not all_texts:
        logger.warning("⚠️ No text extracted from documents")
        return
    
    logger.info(f"📊 Total chunks to index: {len(all_texts)}")
    
    # 6. Generate embeddings
    logger.info("🧠 Generating embeddings...")
    try:
        dense_vectors, sparse_vectors = embedder.embed_both(all_texts)
        logger.info(f"   ✅ Dense: {len(dense_vectors)} vectors")
        logger.info(f"   ✅ Sparse: {len(sparse_vectors)} vectors")
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return
    
    # 7. Index in Qdrant
    logger.info("💾 Indexing in Qdrant...")
    try:
        point_ids = qdrant.upsert_batch(
            texts=all_texts,
            dense_vectors=dense_vectors.tolist() if len(dense_vectors) > 0 else [],
            metadata_list=all_metadata,
        )
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")
        return
    
    # 8. Summary
    count = qdrant.count_points()
    logger.info("=" * 60)
    logger.info("🎉 INDEXING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📊 Files processed: {total_files}")
    logger.info(f"📊 Total chunks indexed: {count}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        index_documents()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()