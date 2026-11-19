"""Tests de l'environnement : ils échouent si une bibliothèque manque ou a changé d'interface."""

import numpy as np


def test_imports_du_rag():
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_openrouter import ChatOpenRouter
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    assert all([FAISS, OpenAIEmbeddings, ChatOpenRouter, RecursiveCharacterTextSplitter])


def test_faiss_retrouve_le_plus_proche_voisin():
    import faiss

    vecteurs = np.eye(4, dtype=np.float32)  # 4 vecteurs bien distincts
    index = faiss.IndexFlatL2(4)
    index.add(vecteurs)
    _, voisins = index.search(vecteurs[2:3], 1)
    assert index.ntotal == 4
    assert voisins[0][0] == 2