# -*- coding: utf-8 -*-
# Scarica testi "puliti" da Wikipedia per base RAG (con fallback via requests)

import os, re, time
from urllib.parse import urlparse
import trafilatura
import requests

URLS = [
    # Forum Catasto
    "https://forum.catasto.it/forum/read/6.html",
    "https://forum.catasto.it/forum/read/2.html%26p%3D17",

    # Immobilio
    "https://www.immobilio.it/threads/visure-catastali-e-piantine.57111/",


    # TopGeometri
    "https://forum.topgeometri.it/t/acquisto-con-riserva-1/3499",
    "https://forum.topgeometri.it/t/richiesta-planimetrie-immobile/8447",

    # GeoLIVE
    "https://www.geolive.org/forum/pregeo-e-docfa/principianti-allo-sbaraglio/atto-notarile-e-suo-deposito-in-conservatoria-immobiliare-30262/"
]



OUTPUT_DIR = "kb_rag_wikipedia"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def safe_slug(url: str) -> str:
    p = urlparse(url)
    last = p.path.rstrip("/").split("/")[-1] or "index"
    last = re.sub(r"[^A-Za-z0-9._-]+", "_", last)
    return f"{p.netloc}__{last}.txt"

def fetch_html(url: str) -> str | None:
    # 1) tentativo con trafilatura
    html = trafilatura.fetch_url(url)
    if html:
        return html
    # 2) fallback via requests
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.ok and r.text:
            return r.text
    except Exception:
        pass
    return None

def extract_text(html: str) -> str | None:
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
        with_metadata=True
    )

def fetch_and_save(url: str) -> str:
    html = fetch_html(url)
    if not html:
        raise RuntimeError("download fallito (fetch_html)")
    text = extract_text(html)
    if not text:
        # Se l’estrazione “pulita” fallisce, salvo almeno l’HTML grezzo come backup
        path_fallback = os.path.join(OUTPUT_DIR, safe_slug(url).replace(".txt", ".html"))
        with open(path_fallback, "w", encoding="utf-8") as f:
            f.write(html)
        raise RuntimeError("estrazione testo fallita (salvato .html di fallback)")
    path = os.path.join(OUTPUT_DIR, safe_slug(url))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

if __name__ == "__main__":
    saved = []
    for u in URLS:
        try:
            fp = fetch_and_save(u)
            print(f"✅ Salvato: {fp}")
            saved.append(fp)
            time.sleep(0.5)  # piccola pausa
        except Exception as e:
            print(f"❌ Errore con {u}: {e}")

    # riepilogo
    if saved:
        print("\nFile creati:")
        for s in saved:
            print(" -", s)
    else:
        print("\nNessun file creato.")
