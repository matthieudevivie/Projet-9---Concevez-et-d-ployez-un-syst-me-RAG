"""Découpe les événements nettoyés en chunks, les vectorise avec mistral-embed et construit l'index Faiss."""

import argparse
import json
import time
from datetime import datetime

from puls_events_rag.chemins import COLLECTE, EVENEMENTS, INDEX
from puls_events_rag.documents import CHEVAUCHEMENT, TAILLE_CHUNK, decouper
from puls_events_rag.index import construire_index, sauvegarder
from puls_events_rag.modeles import creer_embeddings

PRIX_PAR_MILLION = 0.10  # $ par million de jetons pour mistral-embed (OpenRouter, novembre 2026)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, help="n'indexer que les N premiers événements (essai)")
    parser.add_argument("--estimer", action="store_true", help="afficher le découpage et le coût, sans appeler l'API")
    args = parser.parse_args()

    with EVENEMENTS.open(encoding="utf-8") as f:
        evenements = [json.loads(ligne) for ligne in f]
    if args.limite:
        evenements = evenements[:args.limite]
    documents = decouper(evenements)
    caracteres = sum(len(d.page_content) for d in documents)
    jetons = caracteres / 3.5  # ordre de grandeur pour du français ; le vrai compte est sur OpenRouter
    print(f"{len(evenements)} événements -> {len(documents)} chunks, {caracteres:,} caractères")
    print(f"coût estimé : ≈ {jetons / 1e6:.2f} M jetons ≈ {jetons / 1e6 * PRIX_PAR_MILLION:.2f} $")
    if args.estimer:
        return

    debut = time.perf_counter()
    embeddings = creer_embeddings()
    index = construire_index(documents, embeddings)
    duree = time.perf_counter() - debut
    sauvegarder(index, INDEX)
    infos = {
        "construit_le": datetime.now().isoformat(timespec="seconds"),
        "date_ref_collecte": json.loads(COLLECTE.read_text(encoding="utf-8"))["date_ref"],
        "modele": embeddings.model,
        "evenements": len(evenements),
        "chunks": len(documents),
        "taille_chunk": TAILLE_CHUNK,
        "chevauchement": CHEVAUCHEMENT,
        "dimension": index.index.d,
        "duree_s": round(duree, 1),
    }
    (INDEX / "infos.json").write_text(json.dumps(infos, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"index écrit dans {INDEX} en {duree:.0f} s ({index.index.ntotal} vecteurs de dimension {index.index.d})")


if __name__ == "__main__":
    main()