"""Pose des questions à l'index Faiss et affiche les événements les plus proches (sans LLM)."""

import argparse
import time

from puls_events_rag.chemins import INDEX
from puls_events_rag.index import charger_index
from puls_events_rag.modeles import creer_embeddings

QUESTIONS = [
    "Un concert de jazz à Nantes",
    "Une exposition de peinture au musée",
    "Une activité pour des enfants de 6 ans pendant les vacances",
    "Un spectacle d'humour ou de café-théâtre",
    "Une visite du patrimoine pendant les Journées européennes du patrimoine",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="tes questions (sinon, une liste de questions de test)")
    parser.add_argument("-k", type=int, default=4, help="nombre de résultats par question")
    args = parser.parse_args()

    index = charger_index(INDEX, creer_embeddings())
    for question in args.questions or QUESTIONS:
        debut = time.perf_counter()
        resultats = index.similarity_search_with_score(question, k=args.k)
        duree = (time.perf_counter() - debut) * 1000
        print(f"\n« {question} »  ({duree:.0f} ms, appel d'embedding compris)")
        for document, distance in resultats:
            m = document.metadata
            print(f"  {distance:.3f}  {m['titre'][:60]}  |  {m['lieu'][:30]}  |  {m['horaires'] or m['debut'][:10]}")


if __name__ == "__main__":
    main()