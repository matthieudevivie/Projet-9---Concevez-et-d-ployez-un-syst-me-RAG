"""D'un événement nettoyé aux chunks à vectoriser : du texte, et des métadonnées à côté."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

TAILLE_CHUNK = 1000  # caractères de description par chunk (l'en-tête s'y ajoute)
CHEVAUCHEMENT = 150  # caractères répétés d'un chunk au suivant


def entete(ev: dict) -> str:
    """Ce qui identifie l'événement ; répété en tête de chacun de ses chunks."""
    lignes = [ev["titre"], f"Lieu : {ev['lieu']}, {ev['adresse']}"]
    if ev.get("horaires"):
        lignes.append(f"Quand : {ev['horaires']}")
    if ev.get("mots_cles"):
        lignes.append("Mots-clés : " + ", ".join(ev["mots_cles"]))
    if ev.get("age_min") is not None:
        lignes.append(f"Âge : à partir de {ev['age_min']} ans")
    if ev.get("statut") and ev["statut"] != "Programmé":
        lignes.append(f"Statut : {ev['statut']}")
    return "\n".join(lignes)


def corps(ev: dict) -> str:
    """Le résumé, puis la description si elle apporte autre chose que le résumé."""
    resume = (ev.get("resume") or "").strip()
    description = (ev.get("description") or "").strip()
    if description.startswith(resume):  # souvent, la description commence par le résumé
        return description
    return f"{resume}\n{description}".strip()


def metadonnees(ev: dict) -> dict:
    """Ce qu'on garde à côté du vecteur : pour citer, filtrer et afficher, pas pour chercher."""
    garder = ["uid", "titre", "debut", "fin", "horaires", "lieu", "adresse", "url", "statut", "age_min", "agenda"]
    return {cle: ev.get(cle) for cle in garder}


def decouper(evenements: list[dict], taille: int = TAILLE_CHUNK,
             chevauchement: int = CHEVAUCHEMENT) -> list[Document]:
    """Un ou plusieurs chunks par événement, chacun précédé de l'en-tête de l'événement."""
    decoupeur = RecursiveCharacterTextSplitter(chunk_size=taille, chunk_overlap=chevauchement)
    documents = []
    for ev in evenements:
        tete = entete(ev)
        morceaux = decoupeur.split_text(corps(ev)) or [""]  # au moins un chunk, même sans texte
        for numero, morceau in enumerate(morceaux):
            texte = f"{tete}\n\n{morceau}".strip()
            meta = metadonnees(ev) | {"chunk": numero, "chunks": len(morceaux)}
            documents.append(Document(page_content=texte, metadata=meta))
    return documents