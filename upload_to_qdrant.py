# -*- coding: utf-8 -*-
from pathlib import Path
import json, uuid, numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

EMB_DIR    = Path("data_embeddings_v4")
MANIFEST   = EMB_DIR / "manifest_embeddings.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "kb_legale_it"
BATCH_SIZE = 256  # puoi ridurre se hai poca RAM

def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest non trovato: {MANIFEST}")

    client = QdrantClient(url=QDRANT_URL)

    batch = []
    total = 0

    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Upload Qdrant"):
            rec = json.loads(line)
            emb_path = Path(rec["embedding_path"])
            if not emb_path.exists():
                continue

            vec = np.load(emb_path)
            # assicurati che sia lista di float (non numpy types)
            vector = vec.astype(np.float32).tolist()

            payload = {
                "chunk_path": rec.get("chunk_path"),
                "source_path": rec.get("source_path"),
                "source_name": rec.get("source_name"),
                "source_dir":  rec.get("source_dir"),
                "chunk_index": rec.get("chunk_index"),
                "chunk_size_chars": rec.get("chunk_size_chars"),
                "model_name":  rec.get("model_name"),
                "task_label":  rec.get("task_label"),
            }

            batch.append(PointStruct(
                id=str(uuid.uuid4()),   # 👈 ID valido
                vector=vector,
                payload=payload,
            ))

            if len(batch) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION, points=batch)
                total += len(batch)
                batch = []

    # flush ultimo batch
    if batch:
        client.upsert(collection_name=COLLECTION, points=batch)
        total += len(batch)

    print(f"\n✅ Upload completato. Chunk inseriti: {total}")

if __name__ == "__main__":
    main()
