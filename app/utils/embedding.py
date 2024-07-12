import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

def create_embeddings(all_text_segments, client):
    if not all_text_segments:
        logging.error("Não há segmentos de texto válidos para criar embeddings.")
        raise ValueError("Não há segmentos de texto válidos para criar embeddings.")
    
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(all_text_segments)
    logging.info("Embeddings criados com sucesso.")
    return vectorizer, embeddings

def search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments):
    query_vec = vectorizer.transform([prompt])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    best_match_index = np.argmax(similarities)
    best_match_segment = all_text_segments[best_match_index]
    logging.info(f"Melhor correspondência encontrada no índice: {best_match_index}")
    return best_match_index, best_match_segment


def try_fallback_search(prompt, vectorizer, embeddings, all_text_segments):
    try:
        logging.info("Tentando buscar novamente após recriar embeddings.")
        vectorizer, embeddings = create_embeddings(all_text_segments, client=None)
        best_match_index, best_match_segment = search_with_embeddings(prompt, vectorizer, embeddings, all_text_segments)
        return best_match_index, best_match_segment
    except Exception as e:
        logging.error(f"Erro ao tentar a busca de fallback: {e}")
        return None, "Desculpe, não consegui encontrar uma resposta para sua pergunta."