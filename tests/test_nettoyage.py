"""Tests unitaires du nettoyage : de petits événements écrits à la main, sans réseau."""

from datetime import date

import pandas as pd
import pytest

from puls_events_rag.nettoyage import html_vers_texte, libelle, nettoyer
from puls_events_rag.openagenda import construire_filtre

DATE_REF = date(2026, 11, 20)
PROGRAMME = '{"id": 1, "label": {"fr": "Programmé"}}'
ANNULE = '{"id": 6, "label": {"fr": "Annulé"}}'
SUR_PLACE = '{"id": 1, "label": {"fr": "Sur place"}}'
EN_LIGNE = '{"id": 2, "label": {"fr": "En ligne"}}'


def evenement(uid, **modifs):
    """Un événement valide à Nantes ; chaque test change seulement ce qu'il vérifie."""
    base = {
        "uid": uid, "canonicalurl": f"https://openagenda.com/e/{uid}",
        "title_fr": f"Concert {uid}", "description_fr": "Un concert.",
        "longdescription_fr": "<p>Un <b>concert</b>.</p>", "conditions_fr": None,
        "keywords_fr": ["concert"], "daterange_fr": "Samedi 22 novembre, 20h30",
        "firstdate_begin": "2026-11-22T19:30:00+00:00",
        "lastdate_end": "2026-11-22T22:00:00+00:00",
        "location_name": "Pannonica", "location_address": "9 rue Basse Porte, Nantes",
        "location_city": "Nantes", "location_insee": "44109",
        "attendancemode": SUR_PLACE, "status": PROGRAMME,
        "age_min": None, "age_max": None, "originagenda_title": "Nantes Métropole",
    }
    return base | modifs


def uids_gardes(*evenements):
    propres, _ = nettoyer(pd.DataFrame(list(evenements)), DATE_REF)
    return set(propres["uid"])


def test_html_vers_texte():
    assert html_vers_texte("<p>Concert&nbsp;de <b>jazz</b></p><p>Samedi</p>") == "Concert de jazz\nSamedi"
    assert html_vers_texte("<p>Un <b>concert</b>.<br>Gratuit</p>") == "Un concert.\nGratuit"
    assert html_vers_texte(None) == ""


def test_html_vers_texte_resiste_au_html_piege():
    """Ce qu'une regex ne sait pas faire : un > dans un attribut, un commentaire, un script."""
    piege = '<p><a title="1 > 0">Concert</a><!-- note interne --><script>alert(1)</script></p>'
    assert html_vers_texte(piege) == "Concert"


def test_libelle():
    assert libelle(ANNULE) == "Annulé"
    assert libelle(None) is None


def test_filtre_api_remonte_d_un_an():
    assert "lastdate_end >= date'2025-11-20'" in construire_filtre(DATE_REF)


def test_garde_nantes_meme_sans_code_insee():
    assert uids_gardes(evenement("a", location_insee=None, location_city="NANTES ")) == {"a"}


def test_ecarte_une_autre_commune_meme_si_la_ville_dit_nantes():
    assert uids_gardes(evenement("a", location_insee="44026")) == set()


@pytest.mark.parametrize("fin, garde", [
    ("2025-11-19T10:00:00+00:00", False),  # terminé il y a un an et un jour
    ("2025-11-21T10:00:00+00:00", True),  # terminé il y a un peu moins d'un an
    ("2027-03-01T10:00:00+00:00", True),  # à venir
])
def test_periode_d_un_an(fin, garde):
    assert uids_gardes(evenement("a", firstdate_begin=fin, lastdate_end=fin)) == ({"a"} if garde else set())


def test_ecarte_annules_en_ligne_et_agendas_non_culturels():
    gardes = uids_gardes(
        evenement("ok"),
        evenement("annule", status=ANNULE),
        evenement("en_ligne", attendancemode=EN_LIGNE),
        evenement("emploi", originagenda_title="Mes événements France Travail"),
    )
    assert gardes == {"ok"}


def test_supprime_les_doublons_et_compte_dans_le_bilan():
    propres, bilan = nettoyer(pd.DataFrame([
        evenement("a"), evenement("b", title_fr="Concert a"),  # même titre, date, lieu
    ]), DATE_REF)
    assert list(propres["uid"]) == ["a"]
    assert bilan["doublon"] == 1 and bilan["gardés"] == 1


def test_colonnes_de_sortie():
    propres, _ = nettoyer(pd.DataFrame([evenement("a")]), DATE_REF)
    assert propres.loc[0, "description"] == "Un concert."
    assert {"uid", "titre", "description", "debut", "fin", "lieu", "url"} <= set(propres.columns)