"""Contrôles des fichiers produits par les scripts : les événements nettoyés, puis l'index.

Ils vérifient les vraies données : sautés si les fichiers n'existent pas (en CI, par exemple).
"""

import json

import pandas as pd
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from puls_events_rag.chemins import COLLECTE, EVENEMENTS, INDEX
from puls_events_rag.index import charger_index, uids_indexes

sans_evenements = pytest.mark.skipif(not EVENEMENTS.exists(), reason="lancer d'abord scripts/collecter_evenements.py")
sans_index = pytest.mark.skipif(not (INDEX / "index.faiss").exists(), reason="lancer d'abord scripts/construire_index.py")


@pytest.fixture(scope="module")
def evenements():
    df = pd.read_json(EVENEMENTS, lines=True, dtype={"uid": str})
    for colonne in ["debut", "fin"]:
        df[colonne] = pd.to_datetime(df[colonne], utc=True)
    return df


@pytest.fixture(scope="module")
def index():
    # Pour compter et lire ce qui est indexé, pas besoin du vrai modèle (ni de clé) :
    # un faux modèle de la même dimension suffit à recharger l'index.
    return charger_index(INDEX, DeterministicFakeEmbedding(size=1024))


@sans_evenements
def test_assez_d_evenements(evenements):
    assert len(evenements) > 1000


@sans_evenements
def test_identifiants_uniques(evenements):
    assert evenements["uid"].is_unique


@sans_evenements
def test_champs_essentiels_remplis(evenements):
    for colonne in ["titre", "resume", "debut", "fin", "lieu", "url"]:
        assert evenements[colonne].notna().all(), colonne


@sans_evenements
def test_aucun_evenement_annule(evenements):
    assert (evenements["statut"] != "Annulé").all()


@sans_evenements
def test_dates_coherentes(evenements):
    assert (evenements["debut"] <= evenements["fin"]).all()


@sans_evenements
def test_periode_moins_d_un_an(evenements):
    collecte = json.loads(COLLECTE.read_text(encoding="utf-8"))
    limite = pd.Timestamp(collecte["date_ref"], tz="UTC") - pd.Timedelta(days=365)
    assert (evenements["fin"] >= limite).all()


@sans_index
def test_tous_les_evenements_sont_indexes(evenements, index):
    infos = json.loads((INDEX / "infos.json").read_text(encoding="utf-8"))
    if infos["evenements"] != len(evenements):
        pytest.skip(f"index d'essai ({infos['evenements']} événements) : relancer sans --limite")
    assert uids_indexes(index) == set(evenements["uid"])


@sans_index
def test_index_coherent_avec_ses_infos(index):
    infos = json.loads((INDEX / "infos.json").read_text(encoding="utf-8"))
    assert index.index.ntotal == infos["chunks"] == len(index.index_to_docstore_id)
    assert index.index.d == infos["dimension"] == 1024