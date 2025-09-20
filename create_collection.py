from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name="kb_legale_it",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

print("✅ Collection 'kb_legale_it' creata")
