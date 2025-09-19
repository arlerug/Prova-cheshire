from cat.mad_hatter.decorators import hook
from cat.plugins.utils import set_cat_var
import json
import re

SCHEMA_GUIDE = """
Sei un esperto di AI/ML e MLOps. Valuta quanto emerge dal MESSAGGIO UTENTE.
Rispondi SOLO in JSON valido, senza testo extra, con questo schema:

{
  "capabilities_known": ["breve voce 1", "breve voce 2"],
  "concepts_unknown": ["breve voce 1", "breve voce 2"],
  "misconceptions": ["breve voce 1", "breve voce 2"],
  "seniority_guess": "novizio" | "intermedio" | "esperto" | "incerto",
  "confidence": 0.0-1.0
}

Regole:
- Le voci devono essere sintetiche (2-5 parole), es. "RAG con Qdrant", "fine-tuning LoRA".
- Inserisci in "concepts_unknown" SOLO se il messaggio suggerisce esplicitamente mancanza/ richiesta di definizione.
- "misconceptions" se noti termini usati in modo improprio o affermazioni probabilmente errate.
- "seniority_guess": scegli il livello più plausibile in base al messaggio; usa "incerto" se il testo non basta.
- "confidence": stima soggettiva (0..1).
- NON aggiungere testo fuori dal JSON.
"""

def _ask_llm(cat, user_text: str) -> dict:
    prompt = f"{SCHEMA_GUIDE}\n\nMESSAGGIO UTENTE:\n{user_text}\n"
    raw = cat.llm(prompt)
    # Forza estrazione del primo blocco JSON valido
    try:
        # prova parse diretto
        return json.loads(raw)
    except Exception:
        # fallback: cerca un JSON nel testo
        m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # fallback robusto minimale
    return {
        "capabilities_known": [],
        "concepts_unknown": [],
        "misconceptions": [],
        "seniority_guess": "incerto",
        "confidence": 0.0
    }

def _to_instructions(j: dict) -> list[str]:
    lines = []
    for x in j.get("capabilities_known", []) or []:
        lines.append(f"L'utente conosce {x}")
    for x in j.get("concepts_unknown", []) or []:
        lines.append(f"L'utente non sa cos'è {x}")
    for x in j.get("misconceptions", []) or []:
        lines.append(f"L'utente potrebbe avere un fraintendimento su {x}")
    return lines

@hook
def before_cat_reads_message(message, cat):
    """
    - Usa SOLO il LLM per giudicare l'esperienza.
    - Stampa a video una lista di istruzioni tipo:
        "L'utente conosce X", "L'utente non sa cos'è Y", ...
    - Non produce risposte per l'utente; salva i risultati in cat.vars.
    """
    user_text = message.get("text", "")

    judgement = _ask_llm(cat, user_text)
    instructions = _to_instructions(judgement)

    level = judgement.get("seniority_guess", "incerto")
    conf = float(judgement.get("confidence", 0.0) or 0.0)

    # Salva per altri agenti (es. RAG)
    set_cat_var(cat, "user_judgement", judgement)   # JSON completo
    set_cat_var(cat, "user_level", level)           # comodo per riuso
    set_cat_var(cat, "user_level_conf", conf)

    # Stampa a video (stdout/log)
    print("\n[AI Experience Judge]")
    print(f"- livello stimato: {level} (confidence={conf:.2f})")
    if instructions:
        print("- istruzioni derivate:")
        for line in instructions:
            print(f"  • {line}")
    else:
        print("- nessuna istruzione derivabile dal messaggio")

    # NON alteriamo la risposta dell'assistente
    return message

@hook
def before_cat_sends_message(message, cat):
    # Questo agente non modifica le risposte visibili all'utente.
    return message
