import os
import fitz

def extract_text_from_pdf(file):
    """Extrai texto de um arquivo PDF."""
    pdf_data = file.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def split_text(text, max_len=50000):
#     """Divide o texto em segmentos menores."""
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_len:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0
        current_segment.append(word)
        current_length += len(word) + 1
    if current_segment:
        segments.append(' '.join(current_segment))
    return segments


def load_existing_pdfs():
    pdf_paths = [os.path.join("./Data", filename) 
                 for filename in os.listdir("./Data") 
                 if filename.lower().endswith(".pdf")] 
    all_text_segments = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            text = extract_text_from_pdf(file)
            text_segments = split_text(text)
            print("Text")
            print(text)
            print("Text segments")
            print(text_segments)
            all_text_segments.extend(text_segments)
    return all_text_segments