"""Construire, sauvegarder et recharger l'index Faiss, par l'intégration LangChain."""

from collections.abc import Callable
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def construire_index(documents: list[Document], embeddings: Embeddings, taille_lot: int = 500,
                     afficher: Callable[[str], None] = print) -> FAISS:
    """Vectorise les chunks par lots et les ajoute à un index exact (IndexFlatL2)."""
    if not documents:
        raise ValueError("aucun document à indexer")
    index = None
    for debut in range(0, len(documents), taille_lot):
        lot = documents[debut:debut + taille_lot]
        if index is None:
            index = FAISS.from_documents(lot, embeddings)  # crée l'index au premier lot
        else:
            index.add_documents(lot)
        afficher(f"{debut + len(lot)}/{len(documents)} chunks vectorisés")
    return index


def uids_indexes(index: FAISS) -> set[str]:
    """Les identifiants des événements présents dans l'index."""
    return {index.docstore.search(i).metadata["uid"] for i in index.index_to_docstore_id.values()}


def sauvegarder(index: FAISS, dossier: Path) -> None:
    index.save_local(str(dossier))  # écrit index.faiss (les vecteurs) et index.pkl (textes, métadonnées)


def charger_index(dossier: Path, embeddings: Embeddings) -> FAISS:
    # index.pkl est un pickle : on ne recharge que l'index qu'on a construit soi-même.
    return FAISS.load_local(str(dossier), embeddings, allow_dangerous_deserialization=True)