# plugins/intent_router/hooks.py
import json
from cat.mad_hatter.decorators import hook
from cat.log import logger


PROMPT_INTENT = """
Sei un classificatore di richieste relative a documenti notarili e catastali.

Devi decidere SOLO tra due categorie:
1. "info" → l’utente fa domande, vuole chiarimenti, analisi, spiegazioni (es. "Devo controllare i gravami", "Quale documento serve per la conformità").
2. "download" → l’utente chiede direttamente un documento da ottenere/scaricare (es. "Voglio la visura catastale attuale", "Scarica la planimetria").

Rispondi SOLO in JSON nel formato:
{"intent": "info"}
oppure
{"intent": "download"}
"""

def _ask_llm_for_intent(cat, text: str) -> str:
    logger.info("[intent_router] Chiedo al LLM di classificare l'intento...")
    try:
        resp = cat.llm(
            PROMPT_INTENT,
            text,
            temperature=0,
            max_tokens=20
        )
        print("INTENTO")
        data = json.loads(resp.strip())
        intent = data.get("intent", "info")
        if intent not in ["info", "download"]:
            intent = "info"
        return intent
    except Exception as e:
        print("[intent_router] Errore LLM intent:", e)
        return "info"


@hook(priority=5)  # prima del retriever
def before_cat_reads_message(message, cat):
    logger.info("[intent_router] Hook before_cat_reads_message attivato")
    user_text = (message or {}).get("text") or ""
    if not user_text.strip():
        return message

    intent = _ask_llm_for_intent(cat, user_text)

    if not hasattr(cat, "vars") or cat.vars is None:
        cat.vars = {}
    cat.vars["intent"] = intent

    logger.info(f"[intent_router] Intent rilevato dal LLM: {intent}")
    return message
