import os
import fitz
import logging
from concurrent.futures import ProcessPoolExecutor

def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as doc:
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            logging.info(f"Texto extraído de {file_path} com sucesso.")
            return text
    except Exception as e:
        logging.error(f"Erro ao extrair texto de {file_path}: {e}")
        return ""

def process_pdf(pdf_path):
    logging.info(f"Processando arquivo: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if text:
        logging.info(f"Texto extraído (primeiros 100 caracteres): {text[:100]}...")
        return split_text(text)
    else:
        logging.warning(f"Nenhum texto extraído de {pdf_path}")
        return []

def split_text(text):
    words = text.split()
    segments = []
    current_segment = []
    for word in words:
        # Verifica se adicionar a próxima palavra ainda mantém o segmento dentro de um tamanho razoável
        if len(' '.join(current_segment + [word])) > 0:  # Adiciona espaço para as palavras
            segments.append(' '.join(current_segment))
            current_segment = []
        current_segment.append(word)
    
    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments


def load_existing_pdfs(data_dir='../Data'):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pdf_paths = [os.path.join(data_dir, filename)
                 for filename in os.listdir(data_dir)
                 if filename.lower().endswith(".pdf")]

    all_text_segments = []
    with ProcessPoolExecutor() as executor:
        text_segments_list = list(executor.map(process_pdf, pdf_paths))

    for text_segments in text_segments_list:
        all_text_segments.extend(text_segments)

    return all_text_segments
