"""Tests de la chaîne RAG avec de faux modèles : la mécanique (filtre, doublons, sources), sans clé ni coût."""

from datetime import date

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from puls_events_rag.documents import decouper
from puls_events_rag.index import construire_index
from puls_events_rag.prompt import AUCUN_RESULTAT
from puls_events_rag.rag import AssistantRAG, rechercher

AUJOURD_HUI = date(2026, 11, 20)


@pytest.fixture
def index(evenement):
    longue = " ".join(f"Le festival propose le concert numéro {i}." for i in range(80))  # ≈ 3 000 caractères
    evenements = [
        evenement("passe", titre="Concert de jazz passé", fin="2026-10-01T22:00:00.000Z"),
        evenement("jazz", titre="Concert de jazz à venir"),  # fin le 22/11/2026 (fixture de conftest.py)
        evenement("festival", titre="Festival de jazz", description=longue),  # plusieurs chunks
        evenement("expo", titre="Exposition de peinture", fin="2026-12-31T18:00:00.000Z"),
    ]
    return construire_index(decouper(evenements), DeterministicFakeEmbedding(size=64), afficher=lambda _: None)


def texte_de(index, uid):
    """Le texte d'un chunk de l'événement : avec le faux modèle, chercher ce texte le ramène en premier."""
    return next(d.page_content for d in map(index.docstore.search, index.index_to_docstore_id.values())
                if d.metadata["uid"] == uid)


def test_ecarte_les_evenements_termines(index):
    trouves = rechercher(index, texte_de(index, "passe"), AUJOURD_HUI, k=10)
    assert "passe" not in [d.metadata["uid"] for d in trouves]
    assert len(trouves) == 3  # les trois autres événements, terminés après le 20 novembre


def test_un_seul_chunk_par_evenement(index):
    trouves = rechercher(index, texte_de(index, "festival"), AUJOURD_HUI, k=10)
    uids = [d.metadata["uid"] for d in trouves]
    assert uids[0] == "festival" and len(uids) == len(set(uids))
    assert trouves[0].metadata["distance"] == pytest.approx(0, abs=1e-6)


def test_au_plus_k_evenements(index):
    assert len(rechercher(index, "jazz", AUJOURD_HUI, k=2)) == 2


def test_repond_et_rend_les_sources_citees(index):
    llm = FakeListChatModel(responses=["Je vous conseille l'exposition [2]."])
    r = AssistantRAG(index, llm, AUJOURD_HUI, k=3).repondre(texte_de(index, "jazz"))
    assert r["reponse"] == "Je vous conseille l'exposition [2]."
    assert len(r["contextes"]) == 3 and r["contextes"][0].startswith("Concert de jazz à venir")
    assert r["distances"][0] == pytest.approx(0, abs=1e-6) and r["distances"] == sorted(r["distances"])
    assert [s["numero"] for s in r["sources"]] == [2]
    assert r["sources"][0]["uid"] in {"festival", "expo"} and r["sources"][0]["url"].startswith("https://")


def test_sans_evenement_a_venir_le_llm_n_est_pas_appele(index):
    llm = FakeListChatModel(responses=[])  # appelé, il lèverait une erreur
    r = AssistantRAG(index, llm, date(2030, 1, 1)).repondre("un concert")
    assert r["reponse"] == AUCUN_RESULTAT and r["sources"] == [] and r["contextes"] == []


def test_refuse_une_question_vide(index):
    with pytest.raises(ValueError):
        AssistantRAG(index, FakeListChatModel(responses=["x"]), AUJOURD_HUI).repondre("   ")