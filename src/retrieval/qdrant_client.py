"""
Qdrant client wrapper for vector storage and retrieval.
Supports Hybrid Search via RRF (Reciprocal Rank Fusion).
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from src.config import settings
from src.ingestion.embedding import EmbeddingFactory

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """Wrapper for Qdrant vector database operations."""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            prefer_grpc=False,
        )
        self.collection_name = settings.qdrant_collection
        self.embedder = EmbeddingFactory()
        self.dense_dim = self.embedder.get_dense_dimension()
        self.logger = logging.getLogger(f"{__name__}.QdrantClientWrapper")

    def create_collection(self, force: bool = False) -> bool:
        """Create collection with Dense + Sparse vector support."""
        try:
            collections = self.client.get_collections()
            existing = [c.name for c in collections.collections]
        except Exception as e:
            self.logger.error(f"Failed to get collections: {e}")
            existing = []

        if self.collection_name in existing:
            if force:
                self.logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                self.logger.info(f"Collection {self.collection_name} already exists")
                return True

        self.logger.info(f"Creating collection: {self.collection_name}")

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True,
                        ),
                    ),
                },
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                ),
            )
            self.logger.info(f"✅ Collection {self.collection_name} created")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to create collection: {e}")
            raise

    def upsert_batch(
        self,
        texts: List[str],
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Insert or update multiple points in Qdrant."""
        if sparse_vectors is None:
            sparse_vectors = [None] * len(texts)
        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        points = []
        point_ids = []

        for i, (text, dense, sparse, meta) in enumerate(
            zip(texts, dense_vectors, sparse_vectors, metadata_list)
        ):
            point_id = str(uuid4())
            point_ids.append(point_id)

            vectors = {"dense": dense}
            if sparse:
                vectors["sparse"] = models.SparseVector(
                    indices=list(sparse.keys()),
                    values=list(sparse.values()),
                )

            payload = {"text": text, "index": i}
            if meta:
                payload.update(meta)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

        self.logger.info(f"✅ Upserted {len(points)} points")
        return point_ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search using query_points with Prefetch & RRF Fusion."""
        try:
            # 1. Generate embeddings
            dense_query = self.embedder.embed_dense(query)
            if len(dense_query) > 0:
                dense_query = dense_query[0]
            else:
                self.logger.warning("Empty dense query embedding")
                return []

            dense_vector = dense_query.tolist() if hasattr(dense_query, "tolist") else dense_query

            # 2. Build filters
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, (int, float)):
                        conditions.append(FieldCondition(key=key, range=Range(gte=value, lte=value)))
                    else:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                if conditions:
                    query_filter = Filter(must=conditions)

            # 3. Hybrid search via query_points & RRF
            if use_hybrid:
                sparse_query = self.embedder.embed_sparse(query)
                sparse_dict = sparse_query[0] if sparse_query else {}

                prefetch = [
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k * 2,
                        filter=query_filter,
                    ),
                ]

                if sparse_dict:
                    prefetch.append(
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=list(sparse_dict.keys()),
                                values=list(sparse_dict.values()),
                            ),
                            using="sparse",
                            limit=top_k * 2,
                            filter=query_filter,
                        )
                    )

                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            else:
                # Dense-only search
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_vector,
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )

            return [
                {
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                }
                for point in response.points
            ]

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def count_points(self) -> int:
        """Get total number of points in collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            self.logger.error(f"Failed to count points: {e}")
            return 0

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        self.logger.info(f"Collection {self.collection_name} deleted")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}