"""Fixtures partagées par plusieurs fichiers de tests : pytest trouve ce fichier tout seul."""

import pytest


def _evenement(uid="e1", **modifs):
    base = {
        "uid": uid, "titre": "Concert de jazz : trio Lune Bleue", "resume": "Le trio joue ses compositions.",
        "description": "Le trio joue ses compositions. Entrée libre.", "mots_cles": ["jazz", "concert"],
        "horaires": "Samedi 22 novembre, 20h30", "debut": "2026-11-22T19:30:00.000Z",
        "fin": "2026-11-22T22:00:00.000Z", "lieu": "Pannonica", "adresse": "9 rue Basse Porte, Nantes",
        "conditions": None, "age_min": None, "age_max": None, "statut": "Programmé",
        "agenda": "Nantes Métropole", "url": f"https://openagenda.com/e/{uid}",
    }
    return base | modifs


@pytest.fixture
def evenement():
    """Une fabrique : evenement("e2", titre="…") rend un événement valide, modifié seulement là où on le dit."""
    return _evenement