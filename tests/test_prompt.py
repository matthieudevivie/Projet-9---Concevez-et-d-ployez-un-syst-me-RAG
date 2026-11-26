"""Tests du prompt et de la lecture des citations : des fonctions pures, sans modèle."""

from datetime import date

from puls_events_rag.prompt import (
    AUCUN_RESULTAT,
    PROMPT,
    date_en_lettres,
    formater_contexte,
    numeros_cites,
)


def test_date_en_lettres_donne_le_jour_de_la_semaine():
    assert date_en_lettres(date(2026, 11, 20)) == "vendredi 20 novembre 2026"
    assert date_en_lettres(date(2027, 8, 1)) == "dimanche 1 août 2027"


def test_les_extraits_sont_numerotes_a_partir_de_1():
    assert formater_contexte(["Concert", "Expo"]) == "[1] Concert\n\n[2] Expo"


def test_le_prompt_contient_les_consignes_le_contexte_et_la_question():
    messages = PROMPT.invoke({"date": "vendredi 20 novembre 2026", "contexte": "[1] Concert de jazz",
                              "question": "Du jazz ?"}).to_messages()
    systeme, humain = messages
    assert systeme.type == "system" and humain.type == "human"
    assert "vendredi 20 novembre 2026" in systeme.content and "[1] Concert de jazz" in systeme.content
    assert AUCUN_RESULTAT in systeme.content and "N'invente" in systeme.content
    assert humain.content == "Du jazz ?"  # la question reste à part, dans le message de l'utilisateur


def test_numeros_cites_sous_toutes_les_formes():
    assert numeros_cites("Le concert [2] puis l'expo [1].", 5) == [2, 1]
    assert numeros_cites("Deux idées [1, 3], et encore [3][4].", 5) == [1, 3, 4]


def test_numeros_cites_ignore_les_numeros_inventes():
    assert numeros_cites("Voir [7] et [0].", 5) == []
    assert numeros_cites(AUCUN_RESULTAT, 5) == []