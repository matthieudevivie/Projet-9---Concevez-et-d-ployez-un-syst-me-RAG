"""Tests du passage événement -> chunks, sans réseau."""

from puls_events_rag.documents import corps, decouper, entete


def test_entete_contient_titre_lieu_date_et_mots_cles(evenement):
    tete = entete(evenement())
    for attendu in ["Concert de jazz", "Pannonica", "Samedi 22 novembre", "jazz, concert"]:
        assert attendu in tete
    assert "Statut" not in tete  # « Programmé » n'apprend rien


def test_entete_signale_un_statut_inhabituel_et_l_age(evenement):
    tete = entete(evenement(statut="Complet", age_min=6))
    assert "Statut : Complet" in tete and "à partir de 6 ans" in tete


def test_corps_ne_repete_pas_le_resume(evenement):
    assert corps(evenement()) == "Le trio joue ses compositions. Entrée libre."
    assert corps(evenement(description="Autre chose.")) == "Le trio joue ses compositions.\nAutre chose."


def test_un_evenement_court_donne_un_seul_chunk(evenement):
    documents = decouper([evenement()])
    assert len(documents) == 1
    assert documents[0].metadata["uid"] == "e1" and documents[0].metadata["chunks"] == 1


def test_une_longue_description_est_decoupee_et_chaque_chunk_garde_l_entete(evenement):
    longue = " ".join(f"Phrase numéro {i} de la description." for i in range(200))  # ≈ 7 000 caractères
    documents = decouper([evenement(description=longue)], taille=1000, chevauchement=150)
    assert len(documents) > 5
    assert all(d.page_content.startswith("Concert de jazz") for d in documents)
    assert all("Pannonica" in d.page_content for d in documents)
    assert [d.metadata["chunk"] for d in documents] == list(range(len(documents)))


def test_les_metadonnees_ne_gardent_que_l_utile(evenement):
    meta = decouper([evenement()])[0].metadata
    assert {"uid", "titre", "debut", "fin", "lieu", "url"} <= set(meta)
    assert "description" not in meta  # le texte est déjà dans le chunk