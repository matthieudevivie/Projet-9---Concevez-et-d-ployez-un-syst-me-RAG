"""Tests de l'index Faiss avec un faux modèle d'embeddings : sans clé, sans réseau, sans coût."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from puls_events_rag.documents import decouper
from puls_events_rag.index import (
    charger_index,
    construire_index,
    sauvegarder,
    uids_indexes,
)


@pytest.fixture
def embeddings():
    # Même texte -> même vecteur, toujours : les résultats sont reproductibles.
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def documents(evenement):
    titres = ["Concert de jazz", "Exposition de peinture", "Atelier pour enfants", "Visite du château", "Conte"]
    return decouper([evenement(f"e{i}", titre=t) for i, t in enumerate(titres)])


def test_construit_par_lots_sans_perdre_de_chunk(documents, embeddings):
    messages = []
    index = construire_index(documents, embeddings, taille_lot=2, afficher=messages.append)
    assert index.index.ntotal == len(documents) == 5
    assert messages == ["2/5 chunks vectorisés", "4/5 chunks vectorisés", "5/5 chunks vectorisés"]


def test_tous_les_evenements_sont_retrouvables(documents, embeddings):
    index = construire_index(documents, embeddings, afficher=lambda _: None)
    assert uids_indexes(index) == {"e0", "e1", "e2", "e3", "e4"}


def test_un_texte_identique_revient_en_premier(documents, embeddings):
    index = construire_index(documents, embeddings, afficher=lambda _: None)
    cible = documents[3]
    trouve, distance = index.similarity_search_with_score(cible.page_content, k=1)[0]
    assert trouve.metadata["uid"] == "e3" and distance == pytest.approx(0, abs=1e-6)


def test_sauvegarde_puis_rechargement(documents, embeddings, tmp_path):
    sauvegarder(construire_index(documents, embeddings, afficher=lambda _: None), tmp_path)
    assert (tmp_path / "index.faiss").exists() and (tmp_path / "index.pkl").exists()
    recharge = charger_index(tmp_path, embeddings)
    assert recharge.index.ntotal == 5 and uids_indexes(recharge) == {"e0", "e1", "e2", "e3", "e4"}


def test_refuse_une_liste_vide(embeddings):
    with pytest.raises(ValueError):
        construire_index([], embeddings)