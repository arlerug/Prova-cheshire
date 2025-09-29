# -*- coding: utf-8 -*-
from pathlib import Path
import json, numpy as np, torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

CHUNKS_DIR = Path("data_chunks")
EMB_DIR    = Path("data_embeddings_v4")
MODEL_ID   = "jinaai/jina-embeddings-v4"
DEVICE     = "cpu"      # "cuda" se hai GPU
MAX_LEN    = 512        # va bene per chunk ~900 caratteri
TASK_LABEL = "retrieval"  # adapter task: 'retrieval' | 'text-matching' | 'code'

def iter_chunk_files(root: Path):
    for f in root.rglob("*.txt"):
        yield f

def main():
    if not CHUNKS_DIR.exists():
        raise SystemExit(f"Cartella chunk non trovata: {CHUNKS_DIR}")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    manifest_in  = CHUNKS_DIR / "manifest.jsonl"
    manifest_out = EMB_DIR / "manifest_embeddings.jsonl"

    print(f"🔹 Carico {MODEL_ID} …")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float32,   # 👈 forza float32 già in load
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    EMB_DIM = 2048
    (EMB_DIR / "EMBEDDING_DIM.txt").write_text(str(EMB_DIM), encoding="utf-8")
    print(f"✅ Dim embedding: {EMB_DIM}")

    # mappa chunk -> metadati (se presente)
    src_by_chunk = {}
    if manifest_in.exists():
        with manifest_in.open("r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    rec = json.loads(line)
                    src_by_chunk[Path(rec["chunk_path"]).resolve()] = rec
                except Exception:
                    pass

    files = list(iter_chunk_files(CHUNKS_DIR))
    if not files:
        raise SystemExit("Nessun chunk trovato. Esegui prima: python chunk_texts.py")

    total = 0
    with open(manifest_out, "w", encoding="utf-8") as mf_out:
        for f in tqdm(files, desc="Embeddings v4"):
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            # prepara batch testo
            batch = processor.process_texts(texts=[text], prefix=None, max_length=MAX_LEN)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.no_grad():
                out = model.model(**batch, task_label=TASK_LABEL)   # 👈 importante
                vec = out.single_vec_emb                            # (1, 2048)
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                emb = vec[0].to(torch.float32).cpu().numpy()

            # salva npy in cartelle parallele a data_chunks
            rel = f.relative_to(CHUNKS_DIR)
            out_dir = EMB_DIR / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_npy = out_dir / (rel.stem + ".npy")
            np.save(out_npy, emb)

            info = src_by_chunk.get(f.resolve(), {})
            row = {
                "chunk_path": str(f.resolve()),
                "embedding_path": str(out_npy.resolve()),
                "embedding_dim": EMB_DIM,
                "source_path": info.get("source_path"),
                "source_name": info.get("source_name"),
                "source_dir": info.get("source_dir"),
                "chunk_index": info.get("chunk_index"),
                "chunk_size_chars": info.get("chunk_size_chars"),
                "model_name": MODEL_ID,
                "task_label": TASK_LABEL,
                "vector_type": "single",
            }
            mf_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"\nFATTO ✓  Chunk embeddati: {total}")
    print(f"Output: {EMB_DIR} (manifest_embeddings.jsonl, EMBEDDING_DIM.txt)")

if __name__ == "__main__":
    main()
