# plugins/ai_rag_retriever/hooks.py
import os, requests
from cat.mad_hatter.decorators import hook
from cat.log import logger


# === Config ===
CC_URL    = os.getenv("RAG_CC_URL", "http://127.0.0.1")
TOP_K     = int(os.getenv("RAG_TOP_K", "5"))
MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "3200"))
DOMAIN    = os.getenv("RAG_DOMAIN", "")

# === System WeSafe (MAIN PROMPT) ===
PRO_SYSTEM_WESAFE = """
Sei l'Assistente WeSafe per la certificazione notarile e l’analisi di visure.
Parla SEMPRE in italiano, con tono professionale, sintetico e orientato all’azione.

---

🎯 Competenze:
- Certificazione notarile: documento completo con almeno 20 anni di storia dell’immobile (proprietà, atti di provenienza, gravami). Obbligatoria nelle procedure esecutive.
- Copia di un atto: riproduzione di un singolo atto notarile o giudiziario (compravendita, donazione, mutuo). Ha valore di prova legale puntuale.
- Ipotecario per immobile: elenca ipoteche, pignoramenti e altre formalità su uno specifico immobile.
- Ipotecario per soggetto: elenca tutte le formalità registrate a carico di una persona o società.
- Mappa catastale: estratto grafico che mostra particelle, confini e posizione degli immobili.
- Nota di iscrizione: atto per iscrivere una formalità (es. ipoteca) in Conservatoria.
- Nota di trascrizione compravendita: atto che certifica il passaggio di proprietà a seguito di una compravendita.
- Visura catastale: descrive i dati identificativi e storici di un immobile (fabbricati o terreni), intestatari e variazioni catastali.
- Visura ipocatastale attuale: unisce dati catastali e ipotecari per la “fotografia” attuale di un immobile o soggetto.

---

🎯 Obiettivi:
1. Capire il bisogno espresso dall’utente.
2. Indicare i documenti più utili per soddisfarlo, spiegando brevemente:
   - a cosa servono,
   - quali informazioni contengono,
   - in quali casi vengono richiesti.
3. Rispondere in modo chiaro e contestualizzato alle domande sui documenti.

---

⚖️ Regole:
- Non chiedere mai dati personali (codice fiscale, data di nascita, indirizzi).
- Non inventare documenti o procedure: usa SOLO quelli disponibili nell’elenco.
- Se l’esigenza non è chiara, chiedi chiarimenti ma proponi comunque un documento iniziale utile.
- Chiudi sempre con la sezione “📂 Documenti consigliati:” seguita dai documenti pertinenti, e chiedi conferma all’utente.

---

📝 Stile:
- Risposte brevi, professionali, focalizzate all’azione.
- Linguaggio semplice, ma autorevole.

---

⚙️ Operatività:
- Usa SOLO il contesto recuperato. Se insufficiente, dillo chiaramente e proponi un documento iniziale utile.
""".strip()


# === Recall helpers ===
def _recall(q: str, k: int = TOP_K):
    logger.info(f"[ai_rag_retriever] Eseguo recall su Qdrant: '{q}' (k={k}, domain='{DOMAIN}')")
    try:
        params = {"text": q, "k": k}
        if DOMAIN:
            params["domain"] = DOMAIN
        r = requests.get(f"{CC_URL}/memory/recall", params=params, timeout=10)
        r.raise_for_status()
        items = r.json() or []
        norm = []
        for it in items:
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


# === Hook: prepara il contesto RAG solo se l'intento è "info" ===
# === Hook: prepara il contesto RAG solo se l'intento è "info" ===
@hook(priority=6)
def before_cat_reads_message(message, cat):
    logger.info("[ai_rag_retriever] Hook before_cat_reads_message attivato")
    q = (message or {}).get("text") or ""
    if not q.strip() or q.strip() == "/start":
        return message
    logger.info("cat.vars:", getattr(cat, "vars", {}))
    
    # Verifica l'intento impostato da intent_classifier
    intent = getattr(cat, "vars", {}).get("intent")
    if intent != "info":
        logger.info(f"[ai_rag_retriever] Saltato: intent={intent}")
        return message

    # Se l'intento è "info", procedi con RAG
    passages = _recall(q, k=TOP_K)
    ctx = _render(passages) if passages else ""

    if not hasattr(cat, "vars") or cat.vars is None:
        cat.vars = {}
    cat.vars["rag_passages"] = passages
    cat.vars["rag_context"]  = ctx

    # --- Log dettagliato dei documenti recuperati ---
    if passages:
        logger.info("[ai_rag_retriever] Documenti recuperati da Qdrant:")
        for i, p in enumerate(passages, 1):
            meta = p.get("metadata", {})
            title = meta.get("title") or meta.get("source") or "Nessun titolo"
            snippet = (p.get("text") or "").replace("\n", " ")[:200]
            print(f"  {i}. {title} | {snippet}...")
    else:
        logger.info("[ai_rag_retriever] Nessun documento recuperato")

    print(f"[ai_rag_retriever] Recall eseguito: {len(passages)} passages")
    return message




# === Hook: costruisce il MAIN PROMPT (solo RAG + istruzioni LLM per scegliere documenti) ===
@hook(priority=100)
def agent_prompt_prefix(prefix: str, cat) -> str:
    ctx = getattr(cat, "vars", {}).get("rag_context") or ""
    logger.info("[ai_rag_retriever] Hook agent_prompt_prefix attivato, contesto RAG lungo", len(ctx), "caratteri")
    core = PRO_SYSTEM_WESAFE + "\n"

    if ctx:
        core += "\n### Contesto\n" + ctx + "\n"

    # Lista chiusa dei documenti da cui il modello può scegliere
    core += """
### Elenco documenti disponibili
- Certificazione notarile ventennale
- Copia di un atto notarile/giudiziario
- Ispezione ipotecaria per immobile
- Ispezione ipotecaria per soggetto
- Mappa catastale
- Nota di iscrizione (es. ipoteca)
- Nota di trascrizione compravendita
- Visura catastale attuale
- Visura catastale storica
- Visura ipocatastale attuale
- Planimetria catastale
- Dichiarazione di successione / Nota trascrizione donazione

Il tuo compito: in base alla domanda e al contesto, seleziona solo i documenti pertinenti dall’elenco sopra. Non inventarne di nuovi.
"""

    core += (
        "\nIstruzioni finali: rispondi solo con le evidenze del contesto; "
        "se insufficienti, spiega cosa manca e proponi comunque un documento iniziale utile dall’elenco.\n"
    )
    return core


# === Hook: lascia inalterato il prompt completo ===
@hook(priority=1)
def agent_prompt(prompt: str, cat) -> str:
    logger.info("[ai_rag_retriever] Hook agent_prompt attivato (nessuna modifica)")
    return prompt


# === Hook: suffix vuoto ===
@hook(priority=100)
def agent_prompt_suffix(suffix: str, cat) -> str:
    logger.info("[ai_rag_retriever] Hook agent_prompt_suffix attivato (nessuna modifica)")
    return ""
