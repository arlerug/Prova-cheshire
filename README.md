# Cheshire Cat AI – AI Experience Judge

Questo progetto estende [Cheshire Cat](https://github.com/cheshire-cat-ai/core) con un **plugin custom** per classificare il livello di esperienza dell’utente in AI/ML e personalizzare il comportamento del modello.  
L’LLM usato è [Ollama](https://ollama.com/) con il modello `llama3.2:3b`.

---

## 🚀 Funzionalità

- **Plugin `ai_experience_judge`**
  - Analizza ogni messaggio utente con un LLM dedicato
  - Produce un JSON con:
    - concetti noti
    - concetti ignoti
    - possibili fraintendimenti
    - livello stimato (`novizio`, `intermedio`, `esperto`, `incerto`)
  - Stampa i risultati a console
  - Salva tutto in `data/experience_judgements.txt`
  - Espone il livello in `cat.vars` (`user_level`, `user_level_conf`) per altri agenti

- **Prompt professionale**
  - Sostituisce il system prompt predefinito (“gatto di Alice”)
  - Il modello risponde come un **consulente AI/MLOps professionale**
  - Adatta lo stile al livello utente

---

## 📂 Struttura
