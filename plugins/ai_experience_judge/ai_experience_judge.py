from cat.mad_hatter.decorators import hook
import json, re, os
from datetime import datetime

SCHEMA_GUIDE = """
Sei un esperto di • Relazione preliminare o prodromica
  • Trascrizione o iscrizione
  • Certificazione notarile (ex Art. 567 C.P.C.)
  • Visure catastali (storiche per immobile: Fabbricati/Terreni)
  • Atti notarili o giudiziari (atti certi, atti di provenienza)
  • Documenti sui gravami (mutui, ipoteche, formalità pendenti). Valuta quanto emerge dal MESSAGGIO UTENTE.
Rispondi SOLO in JSON valido, senza testo extra, con questo schema:

{
  "capabilities_known": ["breve voce 1", "breve voce 2"],
  "concepts_unknown": ["breve voce 1", "breve voce 2"],
  "misconceptions": ["breve voce 1", "breve voce 2"],
  "seniority_guess": "novizio" | "intermedio" | "esperto" | "incerto",
  "confidence": 0.0-1.0
}
...
"""

LOG_FILE = "/app/cat/data/experience_judgements.txt"

def _ensure_log_ready():
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"# AI Experience Judge log\n# Created: {datetime.now().isoformat()}\n")
        print(f"[AI Experience Judge] Log pronto: {LOG_FILE}")
    except Exception as e:
        print("[AI Experience Judge] Errore creazione log:", e)

_ensure_log_ready()

def _ask_llm(cat, user_text: str) -> dict:
    prompt = f"{SCHEMA_GUIDE}\n\nMESSAGGIO UTENTE:\n{user_text}\n"
    raw = cat.llm(prompt)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {
        "capabilities_known": [],
        "concepts_unknown": [],
        "misconceptions": [],
        "seniority_guess": "incerto",
        "confidence": 0.0
    }

def _to_instructions(j: dict) -> list[str]:
    lines = []
    for x in j.get("capabilities_known") or []:
        lines.append(f"L'utente conosce {x}")
    for x in j.get("concepts_unknown") or []:
        lines.append(f"L'utente non sa cos'è {x}")
    for x in j.get("misconceptions") or []:
        lines.append(f"L'utente potrebbe avere un fraintendimento su {x}")
    return lines

def _append_to_log(user_text: str, judgement: dict, instructions: list[str]):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- {datetime.now().isoformat()} ---\n")
            f.write(f"User text: {user_text}\n")
            f.write("Judgement: " + json.dumps(judgement, ensure_ascii=False) + "\n")
            if instructions:
                f.write("Instructions:\n")
                for line in instructions:
                    f.write(" - " + line + "\n")
    except Exception as e:
        print("[AI Experience Judge] Errore salvataggio log:", e)

@hook(priority=5)
def before_cat_reads_message(message, cat):
    user_text = message.get("text", "")

    judgement = _ask_llm(cat, user_text)
    instructions = _to_instructions(judgement)

    level = judgement.get("seniority_guess", "incerto")
    conf = float(judgement.get("confidence") or 0.0)

    # ⬇️ salva direttamente su cat.vars
    if not hasattr(cat, "vars") or cat.vars is None:
        cat.vars = {}
    cat.vars["user_judgement"] = judgement
    cat.vars["user_level"] = level
    cat.vars["user_level_conf"] = conf

    # console log
    print("\n[AI Experience Judge]")
    print(f"- livello stimato: {level} (confidence={conf:.2f})")
    if instructions:
        print("- istruzioni derivate:")
        for line in instructions:
            print(f"  • {line}")
    else:
        print("- nessuna istruzione derivabile dal messaggio")

    # salva su file
    _append_to_log(user_text, judgement, instructions)

    return message

@hook
def before_cat_sends_message(message, cat):
    return message
