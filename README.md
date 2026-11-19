# Puls-Events : assistant RAG de recommandation d'événements culturels

POC d'un chatbot qui répond aux questions sur les événements culturels à venir,
à partir des données publiques OpenAgenda, avec un système RAG
(Retrieval-Augmented Generation) : LangChain, modèles Mistral (via OpenRouter) et Faiss.

> État : étape 1 (environnement). Les étapes suivantes compléteront ce README.

## Objectifs

- Récupérer et nettoyer les événements OpenAgenda d'une zone, sur moins d'un an.
- Les indexer dans une base vectorielle Faiss, avec leurs métadonnées.
- Répondre aux questions avec un LLM Mistral, orchestré par LangChain.
- Exposer le système par une API REST (FastAPI), conteneurisée avec Docker.
- Mesurer la qualité des réponses sur un jeu de questions annotées.

## Structure du projet

```
scripts/     scripts à lancer (vérification de l'environnement…)
tests/       tests pytest
pyproject.toml, uv.lock   dépendances (uv)
requirements.txt          les mêmes, au format pip (généré)
.env.example              variables d'environnement à renseigner
```

## Installation

Prérequis : Python 3.13, [uv](https://docs.astral.sh/uv/) et une clé d'API OpenRouter
(https://openrouter.ai/settings/keys).

```bash
git clone https://github.com/matthieudevivie/puls-events-rag.git
cd puls-events-rag
uv sync --locked              # installe exactement les versions de uv.lock
cp .env.example .env          # puis renseigner OPENROUTER_API_KEY dans .env
uv run python scripts/check_env.py
uv run pytest
```

Sans uv :

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Notes de version

- LangChain 1.x : `FAISS` s'importe depuis `langchain_community.vectorstores`
  (paquet archivé depuis juin 2026, d'où un `DeprecationWarning` attendu).
- Modèles : Mistral Small 4 (`mistralai/mistral-small-2603`) et `mistral-embed`,
  appelés par OpenRouter (`ChatOpenRouter`, `OpenAIEmbeddings` avec
  `check_embedding_ctx_length=False`), point d'accès européen de Mistral imposé.
  Changer `EMBEDDING_MODEL` impose de reconstruire l'index.
- Après chaque `uv add`, régénérer `requirements.txt` :
  `uv export --format requirements.txt --no-hashes --no-dev -o requirements.txt`