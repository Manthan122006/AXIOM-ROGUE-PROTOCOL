# AXIOM: ROGUE PROTOCOL
## NINE - Ship Intelligence Command (Team 5)

## Start (one command, no extra installs beyond Round 1)
    pip install pandas numpy scikit-learn          # optional: pip install chromadb  -> real Chroma embeddings
    python server.py                               # open http://localhost:8000

Optional environment variables (never put keys in the browser):
    ANTHROPIC_API_KEY=...        live LLM investigation (otherwise clearly labelled "local-demo", NOT an LLM)
    NINE_LLM_MODEL=claude-haiku-5-5
    INBOX_ENDPOINT=http://...    dummy inbox: alerts POSTed on approval (otherwise saved in outbox/)
    NINE_LOGS=hidden.csv         log file to triage (default test.csv)
    NINE_CREW=roster.csv         official crew roster (needs crew_id + module/specialty/role column)
Windows PowerShell:  $env:ANTHROPIC_API_KEY="..."; python server.py

## Pages
/ Command deck (serpent pipeline, run + approve) | /incidents | /agents | /memory | /schedule (download schedule.csv) | /trace

## Batch mode (all logs at once, e.g. hidden-log evaluation)
    python pipeline.py --logs hidden.csv          # predictions.csv, schedule.csv, outbox/, events.js (index.html replay)

## What is real vs fallback
Real: Round 1 classifier (unchanged, threshold 0.38), routing rules, approval gate, schedule.csv with validation, run trace (runs.json).
Fallbacks (labelled in the UI): LLM local-demo without a key; TF-IDF vectors without chromadb; outbox/ files without INBOX_ENDPOINT; placeholder crew without roster.
