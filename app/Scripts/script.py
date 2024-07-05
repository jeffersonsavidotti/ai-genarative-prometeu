import os
import openai
import dotenv
from PyPDF2 import PdfReader
from io import BytesIO
import numpy as np
 
def load_environment_variables():
    dotenv.load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_ID")
    return endpoint, api_key, deployment
 
def setup_openai_client(endpoint, api_key):
    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )
    return client
 
def extract_text_from_pdf(pdf_file):
    pdf = PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page_num in range(len(pdf.pages)):
        text += pdf.pages[page_num].extract_text()
    return text
 
def split_text(text, max_length=3000):
   
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0
 
    for word in words:
        if current_length + len(word) + 1 > max_length:
            segments.append(' '.join(current_segment))
            current_segment = [word]
            current_length = len(word) + 1
        else:
            current_segment.append(word)
            current_length += len(word) + 1
 
    if current_segment:
        segments.append(' '.join(current_segment))
 
    return segments
 
 
def create_embeddings(all_text_segments, client):
    embeddings = []
    for segment in all_text_segments:
        response = client.embeddings.create(input=segment, model="text-embedding-ada-002")
        embeddings.append(response.data[0].embedding)
    return embeddings
 
def search_with_embeddings(query, embeddings, all_text_segments, client):
    response = client.embeddings.create(input=query, model="text-embedding-ada-002")
    query_embedding = response.data[0].embedding
   
    scores = [np.dot(query_embedding, emb) for emb in embeddings]
   
    best_match_index = np.argmax(scores)
    return best_match_index, all_text_segments[best_match_index]