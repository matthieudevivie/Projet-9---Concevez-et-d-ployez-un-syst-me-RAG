"""Accès aux modèles Mistral par OpenRouter."""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter

from puls_events_rag.chemins import RACINE

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELE_EMBEDDINGS = "mistralai/mistral-embed-2312"
MODELE_LLM = "mistralai/mistral-small-2603"  # Mistral Small 4
# Routage B de la séance 01 : Mistral uniquement, sans réutilisation des données.
ROUTAGE = {"order": ["mistral/eu"], "only": ["mistral/eu", "mistral/zdr"], "data_collection": "deny"}


def cle_openrouter() -> str:
    load_dotenv(RACINE / ".env")
    cle = os.getenv("OPENROUTER_API_KEY")
    if not cle:
        raise RuntimeError("OPENROUTER_API_KEY absente : copie .env.example en .env et mets-y ta clé")
    return cle


def creer_embeddings(textes_par_requete: int = 32) -> OpenAIEmbeddings:
    """Le modèle d'embeddings, avec le même routage que le LLM."""
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", MODELE_EMBEDDINGS),
        base_url=OPENROUTER_BASE_URL,
        api_key=cle_openrouter(),
        check_embedding_ctx_length=False,  # envoyer le texte, pas des jetons OpenAI
        chunk_size=textes_par_requete,  # nombre de textes par requête HTTP
        max_retries=6,  # réessaie après une erreur 429 ou 5xx
        model_kwargs={"extra_body": {"provider": ROUTAGE}},
    )

def creer_llm(max_tokens: int = 800) -> ChatOpenRouter:
    """Le LLM qui rédige les réponses, avec le même routage que les embeddings."""
    return ChatOpenRouter(
        model=os.getenv("LLM_MODEL", MODELE_LLM),
        api_key=cle_openrouter(),
        temperature=0,  # la réponse la plus probable : stable d'un appel à l'autre, et sans fantaisie
        max_tokens=max_tokens,  # plafond de la réponse, donc de son coût
        max_retries=3,
        openrouter_provider=ROUTAGE,
    )