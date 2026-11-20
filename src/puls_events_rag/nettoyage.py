"""Nettoyage des événements OpenAgenda : des fonctions pures, testables sans réseau."""

import json
import re
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup

INSEE_NANTES = "44109"

# Agendas sans événement culturel (vérifié sur un échantillon de titres, le 20/11/2026).
AGENDAS_EXCLUS = {
    "Mes événements France Travail",  # recrutements, ateliers emploi
    "Challenges Geovelo",  # défis vélo
    "Maison de la Tranquillité Publique",  # rencontres avec la police municipale
}

# Colonnes gardées, renommées : ce sont les futures métadonnées des chunks.
COLONNES = {
    "uid": "uid",
    "title_fr": "titre",
    "description_fr": "resume",
    "description_longue": "description",
    "keywords_fr": "mots_cles",
    "daterange_fr": "horaires",
    "firstdate_begin": "debut",
    "lastdate_end": "fin",
    "location_name": "lieu",
    "location_address": "adresse",
    "conditions_fr": "conditions",
    "age_min": "age_min",
    "age_max": "age_max",
    "statut": "statut",
    "originagenda_title": "agenda",
    "canonicalurl": "url",
}


# Balises qui terminent un bloc de texte : un retour à la ligne après chacune.
BALISES_BLOC = ["p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"]


def html_vers_texte(contenu: object) -> str:
    """'<p>Concert&nbsp;de <b>jazz</b></p>' -> 'Concert de jazz'."""
    if not isinstance(contenu, str):
        return ""
    soup = BeautifulSoup(contenu, "html.parser")  # un vrai analyseur HTML
    for balise in soup.find_all(["script", "style"]):
        balise.decompose()  # du code, pas du texte : on le retire
    for balise in soup.find_all(["br", "hr"]):
        balise.replace_with("\n")
    for balise in soup.find_all(BALISES_BLOC):
        balise.append("\n")
    texte = soup.get_text()  # le texte seul, entités (&nbsp;, &amp;…) décodées
    texte = re.sub(r"[ \t\xa0]+", " ", texte)  # espaces multiples -> un seul
    texte = re.sub(r"\s*\n\s*", "\n", texte)  # lignes vides en trop
    return texte.strip()


def libelle(valeur: object) -> str | None:
    """'{"id": 6, "label": {"fr": "Annulé", "en": "Cancelled"}}' -> 'Annulé'."""
    if not isinstance(valeur, str):
        return None
    return json.loads(valeur)["label"]["fr"]


def est_a_nantes(df: pd.DataFrame) -> pd.Series:
    """Le code INSEE fait foi ; s'il manque, on se rabat sur le nom de ville."""
    insee = df["location_insee"]
    ville = df["location_city"].fillna("").str.strip().str.lower()
    return (insee == INSEE_NANTES) | (insee.isna() & ville.str.startswith("nantes"))


def nettoyer(brut: pd.DataFrame, date_ref: date, jours: int = 365):
    """Renvoie (événements propres, bilan) ; le bilan compte les lignes écartées par règle."""
    df = brut.copy()
    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], utc=True)
    df["lastdate_end"] = pd.to_datetime(df["lastdate_end"], utc=True)
    df["statut"] = df["status"].map(libelle)
    df["mode"] = df["attendancemode"].map(libelle)
    df["description_longue"] = df["longdescription_fr"].map(html_vers_texte)
    df[["age_min", "age_max"]] = df[["age_min", "age_max"]].astype("Int64")  # 55, pas 55.0
    limite = pd.Timestamp(date_ref, tz="UTC") - pd.Timedelta(days=jours)

    regles = [  # (raison, fonction qui repère les lignes à écarter)
        ("hors de Nantes", lambda d: ~est_a_nantes(d)),
        ("terminé depuis plus d'un an", lambda d: d["lastdate_end"] < limite),
        ("annulé", lambda d: d["statut"] == "Annulé"),
        ("en ligne uniquement", lambda d: d["mode"] == "En ligne"),
        ("agenda non culturel", lambda d: d["originagenda_title"].isin(AGENDAS_EXCLUS)),
        ("doublon", lambda d: d.duplicated(["title_fr", "firstdate_begin", "location_name"])),
    ]
    bilan = {"téléchargés": len(df)}
    df = df.sort_values(["firstdate_begin", "uid"])  # ordre stable : le même doublon est gardé
    for raison, a_ecarter in regles:
        masque = a_ecarter(df)
        bilan[raison] = int(masque.sum())
        df = df[~masque]
    bilan["gardés"] = len(df)
    return df[list(COLONNES)].rename(columns=COLONNES).reset_index(drop=True), bilan