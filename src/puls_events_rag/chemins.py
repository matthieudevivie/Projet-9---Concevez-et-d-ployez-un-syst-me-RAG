"""Chemins du projet, indépendants du dossier d'où l'on lance les commandes."""

import os
from pathlib import Path

# chemins.py est dans src/puls_events_rag/ : on remonte de deux crans pour trouver la racine.
RACINE = Path(__file__).resolve().parents[2]

# PULS_DATA_DIR permettra de placer les données ailleurs (dans un conteneur, par exemple).
DONNEES = Path(os.getenv("PULS_DATA_DIR", RACINE / "data"))
EVENEMENTS = DONNEES / "evenements_nantes.jsonl"
COLLECTE = DONNEES / "collecte.json"
INDEX = DONNEES / "index_faiss"