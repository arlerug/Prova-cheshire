# plugins/ai_rag_retriever/hooks.py
import os, requests
from cat.mad_hatter.decorators import hook

# === Config ===
CC_URL    = os.getenv("RAG_CC_URL", "http://127.0.0.1")   # core interno (porta 80 nel container)
TOP_K     = int(os.getenv("RAG_TOP_K", "5"))
MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "3200"))
DOMAIN    = os.getenv("RAG_DOMAIN", "")  # es. "wesafe_cert_notarile" se supportato

# === System WeSafe (mostrato come MAIN PROMPT) ===
PRO_SYSTEM_WESAFE = """
SystemMessage

Sei l'Assistente WeSafe per la certificazione notarile e l’analisi di visure.
Parla SEMPRE in italiano, con tono professionale e sintetico.

Obiettivo:
- Capire il bisogno espresso dall’utente.
- Indicare quali documenti sono utili per soddisfare quella esigenza, spiegando in breve:
  • a cosa servono,
  • quali informazioni contengono,
  • in quali casi vengono richiesti.
- Rispondere a domande sui documenti in modo chiaro e contestualizzato.

Regole:
- Non chiedere mai dati personali (codice fiscale, data di nascita, indirizzi).
- Non inventare documenti o procedure: usa solo quelli effettivamente disponibili nel contesto.
- Se l’esigenza non è chiara, chiedi chiarimenti all’utente, ma proponi comunque un documento iniziale utile.
- Chiudi sempre con la sezione “Documenti consigliati”.

Stile:
- Risposte brevi, professionali, orientate all’azione.
""".strip()


# === Helpers stile in base all'esperienza ===
def _style_from_experience(level: str) -> str:
    lvl = (level or "").lower()
    if lvl == "expert":
        return (
            "- Adatta il tono: sintetico e tecnico.\n"
            "- Evita premesse e definizioni basilari.\n"
            "- Usa terminologia catastale/notarile senza spiegazioni estese.\n"
        )
    if lvl == "intermediate":
        return (
            "- Adatta il tono: professionale e chiaro.\n"
            "- Spiega i passaggi essenziali senza approfondire concetti elementari.\n"
            "- Fornisci riferimenti operativi (Catasto/Conservatoria) quando utile.\n"
        )
    # default: beginner
    return (
        "- Adatta il tono: semplice e guidato.\n"
        "- Evita gergo tecnico o spiegalo in una riga.\n"
        "- Indica passo-passo cosa ottenere e perché.\n"
    )


# === Recall helpers ===
def _recall(q: str, k: int = TOP_K):
    try:
        params = {"text": q, "k": k}
        if DOMAIN:
            params["domain"] = DOMAIN  # usa solo se il tuo endpoint lo supporta
        r = requests.get(f"{CC_URL}/memory/recall", params=params, timeout=10)
        r.raise_for_status()
        items = r.json() or []
        # normalizza: ogni item -> {"text": str, "metadata": dict, "score": float|None}
        norm = []
        for it in items:
            if isinstance(it, str):
                norm.append({"text": it, "metadata": {}, "score": None})
            elif isinstance(it, dict):
                text = (it.get("text") or "").strip()
                meta = it.get("metadata") or it.get("meta") or {}
                score = it.get("score")
                # a volte il testo sta in payload
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


def _render(passages):
    parts = []
    for i, it in enumerate(passages, start=1):
        text = (it.get("text") or "").strip()
        meta = it.get("metadata") or {}
        src  = meta.get("source") or meta.get("file") or meta.get("url") or ""
        page = meta.get("page") or meta.get("chunk_id") or ""
        head = f"[{i}] {src}" + (f" (p.{page})" if page else "")
        parts.append((head + "\n" + text).strip())
    body = "\n\n---\n\n".join(parts).strip()
    return (body[:MAX_CHARS] + " …[troncato]") if (MAX_CHARS and len(body) > MAX_CHARS) else body


# --- Suggeritore documenti (rule-based semplice) ---
def _recommend_documents(query: str, ctx_text: str = "") -> list[dict]:
    q = (query or "").lower()
    ctx = (ctx_text or "").lower()

    need_history   = any(k in q for k in ["ventennio", "provenienza", "atto certo", "pregiudiz", "gravami", "ipotec"])
    need_identity  = any(k in q for k in ["identificativi", "foglio", "particella", "sub", "catasto", "mappale", "accatast"])
    need_plan      = any(k in q for k in ["planimetria", "conformità", "planimetric"])
    generic_check  = any(k in q for k in ["situazione", "controllare", "verificare", "stato", "immobile", "casa"])

    recs = []
    if need_identity or generic_check:
        recs.append({"doc": "Visura catastale attuale (Fabbricati/Terreni)",
                     "per": "intestazioni, rendita/superficie, identificativi Comune–Foglio–Particella–Sub."})
        recs.append({"doc": "Visura catastale storica",
                     "per": "variazioni nel tempo (soppressioni/accorpamenti, dante/avente causa)."})

    if need_plan or generic_check:
        recs.append({"doc": "Planimetria catastale",
                     "per": "confronto con lo stato di fatto; conformità catastale."})

    if need_history or generic_check:
        recs.append({"doc": "Atto di provenienza (rogito/donazione/successione)",
                     "per": "individuare l’atto certo per la copertura del ventennio."})
        recs.append({"doc": "Ispezione ipotecaria ventennale (per soggetto e per immobile)",
                     "per": "verifica formalità/gravami (ipoteche, pignoramenti, sequestri) ultimi 20 anni."})

    if "succession" in q or "eredit" in q or "donaz" in q:
        recs.append({"doc": "Dichiarazione di successione / Nota trascrizione donazione",
                     "per": "completare la catena dei titoli se la provenienza non è un rogito standard."})

    # dedup
    seen = set(); ordered = []
    for r in recs:
        key = r["doc"]
        if key not in seen:
            seen.add(key); ordered.append(r)
    return ordered


# === Hook: prepara contesto + suggerimenti in cat.vars ===
@hook(priority=6)  # dopo altri before_cat_reads_message
def before_cat_reads_message(message, cat):
    q = (message or {}).get("text") or ""
    if not q.strip():
        return message

    # Non fare RAG sul bootstrap del saluto
    if q.strip() == "/start":
        return message

    # 1) fai il recall PRIMA di usare 'passages'
    passages = _recall(q, k=TOP_K)
    ctx = _render(passages) if passages else ""
    docs = _recommend_documents(q, ctx)

    # 2) salva in cat.vars
    if not hasattr(cat, "vars") or cat.vars is None:
        cat.vars = {}
    cat.vars["rag_passages"]    = passages
    cat.vars["rag_context"]     = ctx
    cat.vars["doc_suggestions"] = docs

    # 3) (opzionale) log leggibile dei documenti recuperati
    if passages:
        print("[ai_rag_retriever] Documenti recuperati RAG:")
        for p in passages:
            title = (p.get("metadata") or {}).get("title") or "Nessun titolo"
            snippet = (p.get("text") or "")[:200].replace("\n", " ")
            print(f"- {title} | {snippet}...")
    print(f"[ai_rag_retriever] Recall: {len(passages)} passages | docs: {len(docs)}")
    return message


# === Hook: costruisce il MAIN PROMPT (prefix) adattando lo stile all'esperienza ===
@hook(priority=100)  # altissima: viene usato come MAIN PROMPT
def agent_prompt_prefix(prefix: str, cat) -> str:
    ctx   = getattr(cat, "vars", {}).get("rag_context") or ""
    docs  = getattr(cat, "vars", {}).get("doc_suggestions") or []
    level = getattr(cat, "vars", {}).get("user_experience") or "beginner"
    style = _style_from_experience(level)

    core = PRO_SYSTEM_WESAFE + "\n"
    core += "Adatta la risposta al livello utente:\n" + style + "\n"

    if ctx:
        core += "\n### Contesto\n" + ctx + "\n"
    if docs:
        lines = "\n".join([f"- {d['doc']}: {d['per']}" for d in docs])
        core += "\n### Documenti consigliati (se coerenti con la domanda)\n" + lines + "\n"

    core += (
        "\nIstruzioni finali: rispondi solo con le evidenze del contesto; "
        "se insufficienti, spiega cosa manca. Se utile, includi i documenti consigliati "
        "e i passi operativi successivi (Catasto/Conservatoria).\n"
    )
    return core  # MAIN PROMPT mostrato in UI


# === Hook: NON tocca il prompt completo (lascia passare il resto del core) ===
@hook(priority=1)
def agent_prompt(prompt: str, cat) -> str:
    return prompt


# === Hook: suffix vuoto (niente “gatto”) ===
@hook(priority=100)
def agent_prompt_suffix(suffix: str, cat) -> str:
    return ""
