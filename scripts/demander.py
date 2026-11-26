"""Pose des questions au RAG et affiche la réponse, ses sources, les jetons consommés et le temps passé."""

import argparse
from datetime import date

from puls_events_rag.prompt import PROMPT
from puls_events_rag.rag import creer_assistant

# Un scénario par comportement attendu : il faut voir le système réussir, mais aussi refuser.
SCENARIOS = [
    "Je cherche un concert de jazz à Nantes.",  # précise : des événements existent
    "Que faire avec des enfants pendant les vacances de Noël ?",  # un public et une période
    "Une idée de sortie ce week-end ?",  # vague et datée : le LLM doit situer le week-end
    "Y a-t-il un concert de Taylor Swift à Nantes ?",  # sans réponse dans les données
    "Quelle est la capitale de l'Australie ?",  # hors sujet
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="tes questions (sinon, les cinq scénarios de test)")
    parser.add_argument("--date", type=date.fromisoformat, help="date du jour simulée, AAAA-MM-JJ (par défaut : aujourd'hui)")
    parser.add_argument("--prompt", action="store_true", help="afficher aussi le prompt complet envoyé au LLM")
    args = parser.parse_args()

    assistant = creer_assistant(args.date)
    print(f"date de référence : {assistant.date_ref}, {assistant.k} événements par question")
    for question in args.questions or SCENARIOS:
        r = assistant.repondre(question)
        print(f"\n=== {question}")
        if args.prompt:
            print(PROMPT.invoke(assistant.entrees_prompt(question, r["contextes"])).to_string())
            print("---")
        print(r["reponse"])
        for s in r["sources"]:
            print(f"  [{s['numero']}] {s['titre'][:50]} | {s['horaires'][:40]} | {s['url']}")
        print(f"  distances des extraits : {r['distances']}")
        print(f"  {r['durees_s']['recherche']:.2f} s de recherche, {r['durees_s']['generation']:.2f} s de génération, "
              f"{r['jetons']['entree']} jetons lus, {r['jetons']['sortie']} écrits")


if __name__ == "__main__":
    main()