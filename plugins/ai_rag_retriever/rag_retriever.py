# plugins/ai_rag_retriever/rag_retriever.py
import os, requests
from datetime import datetime

CC_URL    = os.getenv("RAG_CC_URL", "http://127.0.0.1")
TOP_K     = int(os.getenv("RAG_TOP_K", "5"))
MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "3000"))
LOG_FILE  = "/app/cat/data/rag_retriever.log"

def recall(query: str, k: int = TOP_K, timeout: int = 10):
    if not (query or "").strip():
        return []
    try:
        r = requests.get(f"{CC_URL}/memory/recall", params={"text": query, "k": k}, timeout=timeout)
        r.raise_for_status()
        items = r.json() or []
        norm = []
        for it in items:
            print(it)
            if isinstance(it, str):
                norm.append({"text": it, "metadata": {}, "score": None})
            elif isinstance(it, dict):
                text = (it.get("text") or "").strip()
                meta = it.get("metadata") or it.get("meta") or {}
                score = it.get("score")
                if not text and isinstance(it.get("payload"), dict):
                    text = (it["payload"].get("text") or "").strip()
                    meta = it["payload"].get("metadata") or meta
                if not text:
                    text = str(it)
                norm.append({"text": text, "metadata": meta or {}, "score": score})
            else:
                norm.append({"text": str(it), "metadata": {}, "score": None})
        return norm
    except Exception as e:
        print("[ai_rag_retriever] recall error:", e)
        return []

def render(passages):
    parts = []
    for i, it in enumerate(passages, start=1):
        text = (it.get("text") or "").strip()
        meta = it.get("metadata") or {}
        src  = meta.get("source") or meta.get("file") or meta.get("url") or ""
        page = meta.get("page") or meta.get("chunk_id") or ""
        head = f"[{i}] {src}" + (f" (p.{page})" if page else "")
        parts.append((head.strip() + "\n" + text).strip())
    body = "\n\n---\n\n".join(parts).strip()
    return (body[:MAX_CHARS] + " …[troncato]") if (MAX_CHARS and len(body) > MAX_CHARS) else body

def log(msg: str):
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass
