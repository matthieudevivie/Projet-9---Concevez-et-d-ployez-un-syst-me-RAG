"""Télécharge les événements OpenAgenda de Nantes, les nettoie et les enregistre dans data/."""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from puls_events_rag.chemins import COLLECTE, DONNEES, EVENEMENTS
from puls_events_rag.nettoyage import nettoyer
from puls_events_rag.openagenda import telecharger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date-ref", type=date.fromisoformat, default=date.today(),
                        help="date de référence AAAA-MM-JJ (défaut : aujourd'hui)")
    parser.add_argument("--brut", type=Path,
                        help="réutiliser un téléchargement déjà fait au lieu d'appeler l'API")
    args = parser.parse_args()

    DONNEES.mkdir(exist_ok=True)
    if args.brut:
        brut = json.loads(args.brut.read_text(encoding="utf-8"))
    else:
        brut = telecharger(args.date_ref)
        chemin = DONNEES / f"brut_{args.date_ref.isoformat()}.json"
        chemin.write_text(json.dumps(brut, ensure_ascii=False), encoding="utf-8")
        print(f"instantané brut : {chemin}")

    propres, bilan = nettoyer(pd.DataFrame(brut), args.date_ref)
    propres.to_json(EVENEMENTS, orient="records", lines=True, force_ascii=False, date_format="iso")
    collecte = {"date_ref": args.date_ref.isoformat(), "bilan": bilan}
    COLLECTE.write_text(json.dumps(collecte, ensure_ascii=False, indent=2), encoding="utf-8")
    for raison, nombre in bilan.items():
        print(f"{raison:>28} : {nombre}")
    print(f"écrit : {EVENEMENTS} et {COLLECTE}")


if __name__ == "__main__":
    main()