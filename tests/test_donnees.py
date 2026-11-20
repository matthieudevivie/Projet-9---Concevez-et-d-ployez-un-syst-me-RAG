"""Contrôles du fichier produit par scripts/collecter_evenements.py.

Ils vérifient les vraies données : sautés si le fichier n'existe pas (en CI, par exemple).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

FICHIER = Path("data/evenements_nantes.jsonl")
COLLECTE = Path("data/collecte.json")
pytestmark = pytest.mark.skipif(not FICHIER.exists(), reason="lancer d'abord scripts/collecter_evenements.py")


@pytest.fixture(scope="module")
def evenements():
    df = pd.read_json(FICHIER, lines=True)
    for colonne in ["debut", "fin"]:
        df[colonne] = pd.to_datetime(df[colonne], utc=True)
    return df


def test_assez_d_evenements(evenements):
    assert len(evenements) > 1000


def test_identifiants_uniques(evenements):
    assert evenements["uid"].is_unique


def test_champs_essentiels_remplis(evenements):
    for colonne in ["titre", "resume", "debut", "fin", "lieu", "url"]:
        assert evenements[colonne].notna().all(), colonne


def test_aucun_evenement_annule(evenements):
    assert (evenements["statut"] != "Annulé").all()


def test_dates_coherentes(evenements):
    assert (evenements["debut"] <= evenements["fin"]).all()


def test_periode_moins_d_un_an(evenements):
    collecte = json.loads(COLLECTE.read_text(encoding="utf-8"))
    limite = pd.Timestamp(collecte["date_ref"], tz="UTC") - pd.Timedelta(days=365)
    assert (evenements["fin"] >= limite).all()