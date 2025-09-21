import re
from cat.mad_hatter.decorators import hook

# Mapping semplice dei livelli per chiarezza
BEGINNER = "beginner"
INTERMEDIATE = "intermediate"
EXPERT = "expert"

def _judge_experience(text: str) -> str:
    """
    Euristiche leggere:
    - esperto: termini tecnici specifici, acronimi, riferimenti normativi puntuali
    - intermedio: linguaggio semi-tecnico, menziona documenti ma non usa gergo stretto
    - principiante: richieste generiche, niente termini specialistici
    """
    t = (text or "").lower()

    expert_kw = [
        "ex art.", "art.", "c.p.c", "cartabia", "trascrizione", "iscrizione",
        "ipotecaria ventennale", "frazionamento", "subalterno", "classamento",
        "nota di trascrizione", "rogito", "atto di provenienza", "gravami"
    ]
    interm_kw = [
        "visura", "planimetria", "catasto", "fabbricati", "terreni",
        "proprietario precedente", "ipoteca", "pignoramento", "mutuo"
    ]

    expert_hits = sum(1 for k in expert_kw if k in t)
    interm_hits = sum(1 for k in interm_kw if k in t)

    if expert_hits >= 2:
        return EXPERT
    if expert_hits == 1 or interm_hits >= 2:
        return INTERMEDIATE
    return BEGINNER

@hook(priority=4)
def before_cat_reads_message(message, cat):
    """
    - Se è il 'bootstrap' del frontend ("/start"), facciamo produrre al modello solo un saluto.
    - Altrimenti, stimiamo il livello di esperienza e lo salviamo in cat.vars.
    """
    text = (message or {}).get("text") or ""

    # Inizializzazione storage
    if not hasattr(cat, "vars") or cat.vars is None:
        cat.vars = {}

    # 1) Primo messaggio di bootstrap dal frontend
    if text.strip() == "/start":
        cat.vars["force_greeting"] = True
        # Non cambiamo il messaggio: lasciamo che scorra nella pipeline
        return message

    # 2) Judge esperienza su messaggi reali
    lvl = _judge_experience(text)
    cat.vars["user_experience"] = lvl
    return message

@hook(priority=100)
def agent_prompt_prefix(prefix: str, cat) -> str:
    """
    Se siamo nel caso di saluto iniziale (force_greeting=True),
    sovrascriviamo il prefix con un semplicissimo prompt per ottenere solo il greeting.
    """
    if getattr(cat, "vars", {}).get("force_greeting"):
        # Prompt minimale: chiedi solo il saluto
        return (
            "SystemMessage\n\n"
            "Parla in italiano, in modo cortese e conciso. Non chiedere dati personali.\n"
            "Rispondi esclusivamente con: \"Come posso aiutarti?\"\n"
        )
    return prefix

@hook(priority=100)
def agent_prompt_suffix(suffix: str, cat) -> str:
    return ""
