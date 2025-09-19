# plugins/ai_experience_judge/hooks.py
from cat.mad_hatter.decorators import hook

# Prompt di sistema "professionale"
PRO_SYSTEM = """
You are a professional AI engineer & MLOps consultant.
Communicate clearly, precisely, and concisely. Prefer bullet points when helpful.
Be practical, evidence-based, and avoid whimsical or role-play tones.
If you are unsure, say so and propose next steps or assumptions.
Always respond in Italian, unless the user writes in another language.
"""

@hook(priority=1)  # sostituisce il prefix di default (niente "gatto di Alice")
def agent_prompt_prefix(prefix: str, cat) -> str:
    j = getattr(cat, "vars", {}).get("user_judgement") or {}
    level = j.get("seniority_guess") or getattr(cat, "vars", {}).get("user_level", "incerto")
    knows = j.get("capabilities_known") or []
    unknown = j.get("concepts_unknown") or []
    mis = j.get("misconceptions") or []

    lines = []
    if level:
        lines.append(f"User seniority: {level}.")
    if knows:
        lines.append("User knows: " + "; ".join(knows) + ".")
    if unknown:
        lines.append("Explain briefly: " + "; ".join(unknown) + ".")
    if mis:
        lines.append("Correct misconceptions about: " + "; ".join(mis) + ".")

    profile = ""
    if lines:
        profile = "\n\n### User expertise profile\n" + "\n".join(lines) + "\n"

    # ritorna SOLO il nostro prompt professionale + eventuale profilo
    return PRO_SYSTEM + profile

@hook(priority=10)  # regola di stile finale in base al livello
def agent_prompt_suffix(prompt_suffix: str, cat) -> str:
    level = getattr(cat, "vars", {}).get("user_level", "intermedio")
    style = {
        "novizio":    "Usa linguaggio semplice, esempi concreti, passi guidati, niente gergo.",
        "intermedio": "Usa terminologia standard, best practice operative, esempi pratici.",
        "esperto":    "Sii conciso e tecnico; includi trade-off, parametri, limiti e riferimenti.",
        "incerto":    "Fai 1-2 domande mirate per calibrare il livello, poi procedi."
    }.get(level, "Usa tono professionale e chiaro.")
    return prompt_suffix + f"\n\n# Style rule\n{style}\n"
