"""Vérifie que l'environnement du projet est prêt.

Trois contrôles, du plus simple au plus complet :
1. les bibliothèques du RAG s'importent (LangChain, Faiss, OpenRouter) ;
2. Faiss indexe et retrouve des vecteurs ;
3. les modèles Mistral répondent via OpenRouter, si une clé est présente dans .env.

Usage : uv run python scripts/check_env.py
"""

import os
import sys

import numpy as np
from dotenv import load_dotenv

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def verifier_imports() -> None:
    """Importe les briques du RAG avec leurs chemins actuels (LangChain 1.x)."""
    import faiss
    import langchain
    from langchain_community.vectorstores import FAISS  # noqa: F401
    from langchain_openai import OpenAIEmbeddings  # noqa: F401
    from langchain_openrouter import ChatOpenRouter  # noqa: F401
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

    print(f"[ok] imports : Python {sys.version.split()[0]}, faiss {faiss.__version__}, langchain {langchain.__version__}")


def verifier_faiss() -> None:
    """Indexe 100 vecteurs aléatoires et vérifie que le plus proche voisin du vecteur 0 est lui-même."""
    import faiss

    rng = np.random.default_rng(42)
    vecteurs = rng.random((100, 8), dtype=np.float32)
    index = faiss.IndexFlatL2(8)
    index.add(vecteurs)
    distances, voisins = index.search(vecteurs[:1], 3)
    assert index.ntotal == 100, "tous les vecteurs doivent être indexés"
    assert voisins[0][0] == 0 and distances[0][0] == 0, "un vecteur doit être son propre plus proche voisin"
    print(f"[ok] faiss : {index.ntotal} vecteurs indexés, voisins de v0 : {voisins[0].tolist()}")


def verifier_openrouter() -> None:
    """Calcule un embedding et obtient une réponse courte, si OPENROUTER_API_KEY est définie."""
    load_dotenv()
    cle = os.getenv("OPENROUTER_API_KEY")
    if not cle:
        print("[--] openrouter : OPENROUTER_API_KEY absente, test sauté (copie .env.example en .env)")
        return
    from langchain_openai import OpenAIEmbeddings
    from langchain_openrouter import ChatOpenRouter

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "mistralai/mistral-embed-2312"),
        base_url=OPENROUTER_BASE_URL,
        api_key=cle,
        check_embedding_ctx_length=False,  # envoyer le texte, pas des jetons OpenAI
    )
    vecteur = embeddings.embed_query("Concert de jazz ce week-end")
    print(f"[ok] embeddings : dimension {len(vecteur)}")

    llm = ChatOpenRouter(
        model=os.getenv("LLM_MODEL", "mistralai/mistral-small-2603"),
        temperature=0,
        max_tokens=10,
        openrouter_provider={"only": ["mistral/eu"], "allow_fallbacks": False, "data_collection": "deny"},
    )
    reponse = llm.invoke("Réponds par un seul mot : prêt.")
    print(f"[ok] llm : {reponse.content!r}")


if __name__ == "__main__":
    verifier_imports()
    verifier_faiss()
    verifier_openrouter()

mkdir scripts tests              # puis crée scripts/check_env.py avec le code ci-dessus
uv run python scripts/check_env.py