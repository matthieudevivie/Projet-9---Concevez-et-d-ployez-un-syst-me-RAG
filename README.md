# Puls-Events : assistant RAG de recommandation d'événements culturels

POC d'un chatbot qui répond aux questions sur les événements culturels à venir,
à partir des données publiques OpenAgenda, avec un système RAG
(Retrieval-Augmented Generation) : LangChain, modèles Mistral (via OpenRouter) et Faiss.

> État : étape 2 (collecte et nettoyage des données). Les étapes suivantes compléteront ce README.

## Objectifs

- Récupérer et nettoyer les événements OpenAgenda d'une zone, sur moins d'un an.
- Les indexer dans une base vectorielle Faiss, avec leurs métadonnées.
- Répondre aux questions avec un LLM Mistral, orchestré par LangChain.
- Exposer le système par une API REST (FastAPI), conteneurisée avec Docker.
- Mesurer la qualité des réponses sur un jeu de questions annotées.

## Structure du projet

```
src/puls_events_rag/   le code du RAG (collecte, nettoyage…), installé comme paquet
data/                  données téléchargées et nettoyées (non versionnées, voir plus bas)
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
git clone https://github.com/matthieudevivie/Projet-9---Concevez-et-d-ployez-un-syst-me-RAG.git puls-events-rag
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

## Données

Les événements viennent du jeu public OpenAgenda publié par OpenDataSoft
(sans clé d'API). Zone : Nantes (code INSEE 44109) ; période : événements
terminés depuis moins d'un an, en cours ou à venir, par rapport à une date de référence.

```bash
uv run python scripts/collecter_evenements.py --date-ref 2026-11-20
```

Le script écrit `data/brut_<date>.json` (l'instantané brut), `data/evenements_nantes.jsonl`
(un événement nettoyé par ligne) et `data/collecte.json` (date de référence et bilan
du nettoyage). Règles : événements annulés, en ligne uniquement et doublons écartés,
ainsi que trois agendas non culturels (voir `AGENDAS_EXCLUS` dans
`src/puls_events_rag/nettoyage.py`).

## Index vectoriel

Chaque événement devient un ou plusieurs chunks (1 000 caractères de description,
coupés de préférence entre deux paragraphes ; jusqu'à 150 caractères repris d'un chunk
au suivant quand un paragraphe doit être coupé), tous précédés d'un en-tête : titre, lieu, horaires, mots-clés.
Les chunks sont vectorisés avec `mistral-embed` (1 024 dimensions, via OpenRouter)
et indexés dans Faiss (`IndexFlatL2`, recherche exacte), avec leurs métadonnées
(identifiant, dates, lieu, lien).

```bash
uv run python scripts/construire_index.py --estimer
uv run python scripts/construire_index.py
uv run python scripts/tester_recherche.py "un concert de jazz"
```

La première commande affiche le découpage et le coût estimé sans appeler l'API ;
la deuxième construit `data/index_faiss/` (environ 0,26 $ pour Nantes) ;
la troisième affiche les événements les plus proches d'une question.

## Notes de version

- LangChain 1.x : `FAISS` s'importe depuis `langchain_community.vectorstores`
  (paquet archivé depuis juin 2026, d'où un `DeprecationWarning` attendu).
- Modèles : Mistral Small 4 (`mistralai/mistral-small-2603`) et `mistral-embed`,
  appelés par OpenRouter (`ChatOpenRouter`, `OpenAIEmbeddings` avec
  `check_embedding_ctx_length=False`), point d'accès européen de Mistral préféré, repli vers mistral/zdr.
  Changer `EMBEDDING_MODEL` impose de reconstruire l'index.
- Après chaque `uv add`, régénérer `requirements.txt` :
  `uv export --format requirements.txt --no-hashes --no-dev -o requirements.txt`