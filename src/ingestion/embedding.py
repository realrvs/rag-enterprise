import os
from huggingface_hub import login
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory for generating Dense and Sparse embeddings."""
    
    def __init__(self, use_local: bool = True):
        self.logger = logging.getLogger(f"{__name__}.EmbeddingFactory")
        
        # Выбор режима
        self.use_local = use_local

        # В методе __init__ класса EmbeddingFactory:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            self.logger.info("✅ Hugging Face authenticated")
        
        # Dense embedding
        if use_local:
            # Используем локальную модель
            self.logger.info("🧠 Using local dense embedding model...")
            try:
                self.local_model = SentenceTransformer(
                    'intfloat/multilingual-e5-large',
                    device='cpu'  # или 'cuda' если есть GPU
                )
                self.dense_dim = 1024  # Размерность e5-large
                self.logger.info("✅ Local dense model loaded (dim=1024)")
            except Exception as e:
                self.logger.warning(f"⚠️ Local model failed: {e}, falling back to OpenAI")
                self.use_local = False
                self.dense_dim = 1536
                self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            # OpenAI
            self.dense_model = settings.openai_embedding_model
            self.dense_dim = 1536
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Sparse embedding (SPLADE)
        self.sparse_available = False
        self.sparse_model = None
        
        try:
            models_to_try = [
                "Qdrant/bm42-all-minilm-l6-v2-attentions",
                "Qdrant/SPLADE-v2",
            ]
            
            for model_name in models_to_try:
                try:
                    self.sparse_model = SparseTextEmbedding(model_name=model_name)
                    self.sparse_available = True
                    self.sparse_model_name = model_name
                    self.logger.info(f"✅ Sparse embedding model loaded: {model_name}")
                    break
                except Exception:
                    continue
            
            if not self.sparse_available:
                self.logger.warning("⚠️ No SPLADE model available, sparse embeddings disabled")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Sparse embeddings not available: {e}")
            self.sparse_available = False
    
    def embed_dense(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate dense embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        if self.use_local and hasattr(self, 'local_model'):
            try:
                # Локальная модель
                embeddings = self.local_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return np.array(embeddings)
            except Exception as e:
                self.logger.error(f"Local dense embedding failed: {e}")
                # Fallback к нулевым векторам
                return np.zeros((len(texts), self.dense_dim))
        
        # OpenAI
        try:
            response = self.openai_client.embeddings.create(
                model=self.dense_model,
                input=texts,
            )
            embeddings = np.array([e.embedding for e in response.data])
            return embeddings
        except Exception as e:
            self.logger.error(f"OpenAI dense embedding failed: {e}")
            return np.zeros((len(texts), self.dense_dim))
    
    def embed_sparse(self, texts: Union[str, List[str]]) -> List[Dict[int, float]]:
        """Generate sparse embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        if not self.sparse_available or self.sparse_model is None:
            return [{} for _ in texts]
        
        try:
            embeddings = list(self.sparse_model.embed(texts))
            result = []
            for emb in embeddings:
                sparse_dict = dict(zip(emb.indices.tolist(), emb.values.tolist()))
                result.append(sparse_dict)
            return result
        except Exception as e:
            self.logger.error(f"Sparse embedding failed: {e}")
            return [{} for _ in texts]
    
    def embed_both(self, texts: Union[str, List[str]]) -> tuple:
        """Generate both dense and sparse embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        dense = self.embed_dense(texts)
        sparse = self.embed_sparse(texts)
        return dense, sparse
    
    def get_dense_dimension(self) -> int:
        """Get dense embedding dimension."""
        return self.dense_dim