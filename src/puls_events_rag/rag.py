"""Le système RAG : chercher les événements à venir proches de la question, puis faire rédiger la réponse."""

import time
from datetime import date

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from puls_events_rag.chemins import INDEX
from puls_events_rag.index import charger_index
from puls_events_rag.modeles import creer_embeddings, creer_llm
from puls_events_rag.prompt import (
    AUCUN_RESULTAT,
    PROMPT,
    date_en_lettres,
    formater_contexte,
    numeros_cites,
)

NB_EVENEMENTS = 5  # événements distincts donnés au LLM


def rechercher(index: FAISS, question: str, date_ref: date, k: int = NB_EVENEMENTS) -> list[Document]:
    """Les k événements non terminés les plus proches de la question, un chunk (le plus proche) par événement."""
    jour = date_ref.isoformat()

    def pas_termine(meta: dict) -> bool:
        return meta["fin"] >= jour  # dates ISO : l'ordre alphabétique est l'ordre chronologique

    tous = index.index.ntotal  # tout l'index est examiné : le même résultat qu'un pré-filtre, sans réglage à justifier
    resultats = index.similarity_search_with_score(question, k=tous, fetch_k=tous, filter=pas_termine)
    
    retenus = {}
    for doc, distance in resultats:  # du plus proche au plus lointain
        uid = doc.metadata["uid"]
        if uid not in retenus:  # les chunks suivants d'un même événement n'apportent qu'un doublon
            retenus[uid] = Document(page_content=doc.page_content, metadata=doc.metadata | {"distance": float(distance)})
        if len(retenus) == k:
            break
    return list(retenus.values())


def source(numero: int, doc: Document) -> dict:
    """Ce qu'on rend pour citer un événement : de quoi l'afficher et le retrouver."""
    m = doc.metadata
    return {"numero": numero, "uid": m["uid"], "titre": m["titre"], "lieu": m["lieu"],
            "horaires": m["horaires"], "url": m["url"], "distance": round(m["distance"], 3)}


class AssistantRAG:
    """Le système complet : créé une fois (index en mémoire), interrogé autant de fois qu'on veut."""

    def __init__(self, index: FAISS, llm: BaseChatModel, date_ref: date | None = None, k: int = NB_EVENEMENTS):
        self.index = index
        self.k = k
        self.date_ref = date_ref or date.today()
        self.generer = PROMPT | llm  # LCEL : le prompt rempli part au LLM, qui rend un AIMessage

    def entrees_prompt(self, question: str, extraits: list[str]) -> dict:
        return {"date": date_en_lettres(self.date_ref), "contexte": formater_contexte(extraits), "question": question}

    def repondre(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("la question est vide")
        debut = time.perf_counter()
        documents = rechercher(self.index, question, self.date_ref, self.k)
        milieu = time.perf_counter()
        extraits = [d.page_content for d in documents]
        if documents:
            message = self.generer.invoke(self.entrees_prompt(question, extraits))
            reponse, jetons = message.text, message.usage_metadata or {}
        else:  # rien à proposer : on ne laisse pas le LLM improviser, et l'appel ne coûte rien
            reponse, jetons = AUCUN_RESULTAT, {}
        fin = time.perf_counter()
        return {
            "question": question,
            "reponse": reponse,
            "sources": [source(n, documents[n - 1]) for n in numeros_cites(reponse, len(documents))],
            "contextes": extraits,  # tout ce que le LLM a lu : l'évaluation (étape 5) en aura besoin
            "distances": [round(d.metadata["distance"], 3) for d in documents],
            "jetons": {"entree": jetons.get("input_tokens"), "sortie": jetons.get("output_tokens")},
            "durees_s": {"recherche": round(milieu - debut, 3), "generation": round(fin - milieu, 3)},
        }


def creer_assistant(date_ref: date | None = None) -> AssistantRAG:
    """L'assistant branché sur le vrai index et les vrais modèles (clé OpenRouter requise)."""
    return AssistantRAG(charger_index(INDEX, creer_embeddings()), creer_llm(), date_ref)