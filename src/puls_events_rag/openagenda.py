"""Téléchargement des événements OpenAgenda (jeu public publié par OpenDataSoft)."""

from datetime import date, timedelta

import requests

URL_EXPORT = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/evenements-publics-openagenda/exports/json"
)

# Seulement les champs utiles au RAG. Les coordonnées des contributeurs
# (nom, e-mail, téléphone) ne sont jamais téléchargées : minimisation RGPD.
CHAMPS = [
    "uid", "canonicalurl", "title_fr", "description_fr", "longdescription_fr",
    "conditions_fr", "keywords_fr", "daterange_fr", "firstdate_begin", "lastdate_end",
    "location_name", "location_address", "location_postalcode", "location_city",
    "location_insee", "location_coordinates", "attendancemode", "status",
    "age_min", "age_max", "originagenda_title", "updatedat",
]


def construire_filtre(date_ref: date, jours: int = 365) -> str:
    """Filtre large côté API : Nantes, terminé depuis moins de `jours` jours ou à venir.

    Il est volontairement large (le nom de ville est mal saisi) ;
    le filtre précis est fait ensuite par `nettoyage.nettoyer`.
    """
    debut = date_ref - timedelta(days=jours)
    return (
        "(location_insee = '44109' or location_city like 'nantes') "
        f"and lastdate_end >= date'{debut.isoformat()}'"
    )


def telecharger(date_ref: date, timeout: int = 120) -> list[dict]:
    """Télécharge tous les événements du filtre, en une seule requête d'export."""
    params = {"select": ",".join(CHAMPS), "where": construire_filtre(date_ref)}
    reponse = requests.get(URL_EXPORT, params=params, timeout=timeout)
    reponse.raise_for_status()
    return reponse.json()